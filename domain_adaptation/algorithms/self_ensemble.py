from supervised import Supervised
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

class SelfEnsemble(Supervised):
    def __init__(self, encoder, classifier, device):
        super().__init__(encoder, classifier, device)

        self.student_encoder = self.encoder
        self.student_classifier = self.classifier
        self.teacher_encoder = self.encoder
        self.teacher_classifier = self.classifier

        # disbale grad in teacher network
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False
        for param in self.teacher_classifier.parameters():
            param.requires_grad = False

    def train(self, source_dataloader, target_dataloader_student, target_dataloader_teacher, target_dataloader_test, epochs, hyperparams, save_path):    
        # configure hyperparameters
        criterion = nn.CrossEntropyLoss()
        lr_student = hyperparams['learning_rate']
        unsupervised_weight = hyperparams['unsupervised_weight']
        wd = hyperparams['weight_decay']
        cyclic_scheduler = hyperparams['cyclic_scheduler']

        iters = max(len(source_dataloader), len(target_dataloader_student), len(target_dataloader_teacher))

        # configure optimizers and schedulers
        student_optimizer = optim.Adam(list(self.student_encoder.parameters()) + list(self.student_classifier.parameters()), lr=lr_student, weight_decay=wd)
        #teacher_optimizer = EMA(list(self.student_encoder.parameters()) + list(self.student_classifier.parameters()), list(self.teacher_encoder.parameters()) + list(self.teacher_classifier.parameters()))
        teacher_optimizer = self._EMA(self.student_encoder, self.student_classifier, self.teacher_encoder, self.teacher_classifier)
        if cyclic_scheduler:
            scheduler = optim.lr_scheduler.OneCycleLR(student_optimizer, lr_student, epochs=epochs, steps_per_epoch=iters)

        # early stopping variables
        start_epoch = 0
        best_acc = 0.0
        patience = 15
        bad_epochs = 0

        self.history = {'supervised_loss': [],
                        'unsupervised_loss': [],
                        'unsupervised_mask_count': [],
                        'student_accuracy_source': [],
                        'teacher_accuracy_source': [],
                        'student_accuracy_target': [],
                        'teacher_accuracy_target': []
                        }

        # training loop 
        for epoch in range(start_epoch, epochs):
            # set network to training mode
            self.student_encoder.train()
            self.student_classifier.train()
            self.teacher_encoder.train()
            self.teacher_classifier.train()

            running_supervised_loss = 0.0
            running_unsupervised_loss = 0.0
            running_unsupervised_mask_count = 0.0 # best teacher_net is the one with largest unsupervised_mask_count

            for (data_source, labels_source), (data_target_student, _), (data_target_teacher, _) in zip(source_dataloader, target_dataloader_student, target_dataloader_teacher):
                data_source = data_source.to(self.device)
                labels_source = labels_source.to(self.device)
                data_target_student = data_target_student.to(self.device)
                data_target_teacher = data_target_teacher.to(self.device)

                # zero gradients
                student_optimizer.zero_grad()
                
                # classify
                student_source_logits = self.student_classifier(self.student_encoder(data_source))
                student_target_logits = self.student_classifier(self.student_encoder(data_target_student))
                student_target_outputs = F.softmax(student_target_logits, dim=1)

                teacher_target_logits = self.teacher_classifier(self.teacher_encoder(data_target_teacher))
                teacher_target_outputs = F.softmax(teacher_target_logits, dim=1)

                # compute supervised (source) loss
                supervised_loss = criterion(student_source_logits, labels_source)
                
                # compute unsupervised (target) loss
                unsupervised_loss, unsupervised_mask_count = self._augmentation_loss(student_target_outputs, teacher_target_outputs, confidence_thresh=0.9, class_balance=0.01)

                # backpropagate and update weights
                loss = supervised_loss + unsupervised_loss * unsupervised_weight

                loss.backward()
                student_optimizer.step()
                teacher_optimizer.step()
                
                # metrics
                running_supervised_loss += supervised_loss.item()
                running_unsupervised_loss += unsupervised_loss.item()
                running_unsupervised_mask_count += float(unsupervised_mask_count)
                
                # apply scheduler
                if cyclic_scheduler:
                    scheduler.step()

            # get metrics
            epoch_supervised_loss = running_supervised_loss / iters
            epoch_unsupervised_loss = running_unsupervised_loss / iters
            epoch_unsupervised_mask_count = running_unsupervised_mask_count / iters

            self.history['supervised_loss'].append(epoch_supervised_loss)
            self.history['unsupervised_loss'].append(epoch_unsupervised_loss)
            self.history['unsupervised_mask_count'].append(epoch_unsupervised_mask_count)

            # self.evaluate on training data
            student_accuracy_source = self.evaluate(source_dataloader)
            teacher_accuracy_source = self.evaluate(source_dataloader)
            
            student_accuracy_target = self.evaluate(target_dataloader_test)
            teacher_accuracy_target = self.evaluate(target_dataloader_test)        

            self.history['student_accuracy_source'].append(student_accuracy_source)
            self.history['teacher_accuracy_source'].append(teacher_accuracy_source)
            self.history['student_accuracy_target'].append(student_accuracy_target)
            self.history['teacher_accuracy_target'].append(teacher_accuracy_target)
            
            # save checkpoint
            if student_accuracy_target > teacher_accuracy_target:
                curr_acc = student_accuracy_target
            else:
                curr_acc = teacher_accuracy_target
            
            if curr_acc > best_acc:
                torch.save({'student_encoder_weights': self.student_encoder.state_dict(),
                            'student_classifier_weights': self.student_classifier.state_dict(),
                            'teacher_encoder_weights': self.teacher_encoder.state_dict(),
                            'teacher_classifier_weights': self.teacher_classifier.state_dict(),
                        }, save_path)
                best_acc = curr_acc
                bad_epochs = 0
            
            else:
                bad_epochs += 1

            print('[Epoch {}/{}] supervised loss: {:.6f}; unsupervised loss: {:.6f}; unsupervised mask count: {:.6f};'.format(epoch+1, epochs, epoch_supervised_loss, epoch_unsupervised_loss, epoch_unsupervised_mask_count))
            print('[Epoch {}/{}] student accuracy source: {:.6f}; teacher accuracy source: {:.6f}; student accuracy target: {:.6f}; teacher accuracy target: {:.6f};'.format(epoch+1, epochs, student_accuracy_source, teacher_accuracy_source, student_accuracy_target, teacher_accuracy_target))
            
            if bad_epochs >= patience:
                print(f"reached {bad_epochs} bad epochs, stopping training with best val accuracy of {best_acc}!")
                break
                
        best = torch.load(save_path)
        self.student_encoder.load_state_dict(best["student_encoder_weights"])
        self.student_classifier.load_state_dict(best["student_classifier_weights"])
        self.teacher_encoder.load_state_dict(best["teacher_encoder_weights"])
        self.teacher_classifier.load_state_dict(best["teacher_classifier_weights"])

        return self.student_encoder, self.student_classifier, self.teacher_encoder, self.teacher_classifier, self.history

    def plot_metrics(self):
        # plot metrics for losses n stuff
        fig, axs = plt.subplots(1, 3, figsize=(18,5), dpi=200)

        epochs = len(self.history['supervised_loss'])

        axs[0].plot(range(1, epochs+1), self.history['supervised_loss'])
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Supervised loss')

        axs[1].plot(range(1, epochs+1), self.history['unsupervised_loss'])
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Loss')
        axs[1].set_title('Unsupervised loss')

        axs[2].plot(range(1, epochs+1), self.history['unsupervised_mask_count'])
        axs[2].set_xlabel('Epochs')
        axs[2].set_ylabel('Counts')
        axs[2].set_title('Unsupervised mask count')      
            
        plt.show()

        fig, axs = plt.subplots(1, 2, figsize=(12,5), dpi=200)

        axs[0].plot(range(1, epochs+1), self.history['student_accuracy_source'], label='Student')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Accuracy')

        axs[0].plot(range(1, epochs+1), self.history['teacher_accuracy_source'], label='Teacher')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_title('Accuracy on source')

        axs[1].plot(range(1, epochs+1), self.history['student_accuracy_target'], label='Student')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy')

        axs[1].plot(range(1, epochs+1), self.history['teacher_accuracy_target'], label='Teacher')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy')
        axs[1].set_title('Accuracy on target')

        axs[0].legend()
        axs[1].legend()

        plt.show()

    class _EMA(object):
        """
        Exponential moving average weight optimizer for mean teacher model.
        The student network is trained with gradient descent while the weigts of the teacher network are an exponential moving average of the student's.
        """
        def __init__(self, source_encoder, source_classifier, target_encoder, target_classifier, alpha=0.99):        
            # get network parameters (weights)
            source_encoder_params = list(source_encoder.state_dict().values())
            source_classifier_params = list(source_classifier.state_dict().values())
            target_encoder_params = list(target_encoder.state_dict().values())
            target_classifier_params = list(target_classifier.state_dict().values())
            self.alpha = alpha

            self.source_params = source_encoder_params + source_classifier_params
            self.target_params = target_encoder_params + target_classifier_params

            for tgt_p, src_p in zip(self.target_params, self.source_params):
                tgt_p[:] = src_p[:] 

        def step(self):
            one_minus_alpha = 1.0 - self.alpha
            for tgt_p, src_p in zip(self.target_params, self.source_params):
                tgt_p.mul_(self.alpha)
                tgt_p.add_(src_p * one_minus_alpha)

    @staticmethod
    def augmentation_loss(student_output, teacher_output, confidence_thresh=0.96837722, class_balance=0.005, n_classes=3):
        confidence_teacher = torch.max(teacher_output, 1)[0]
        confidence_mask = (confidence_teacher > confidence_thresh).float()
        confidence_mask_count = confidence_mask.sum() # n_samples

        aug_loss = (student_output - teacher_output)**2

        aug_loss = aug_loss.mean(dim=1)
        unsupervised_loss = (aug_loss * confidence_mask).mean()

        # compute class balance loss
        if class_balance > 0.0:
            avg_class_prob = student_output.mean(dim=0) # per-sample average to get average class prediction

            # robust_binary_crossentropy
            inv_tgt = -avg_class_prob + 1.0
            inv_pred = -float(1.0/n_classes) + 1.0 + 1e-6
            equalize_class_loss = -(avg_class_prob * torch.log(float(1.0/n_classes) + 1.0e-6) + inv_tgt * torch.log(inv_pred))

            equalize_class_loss = equalize_class_loss.mean() * n_classes
            equalize_class_loss = equalize_class_loss * confidence_mask.mean(dim=0)

            unsupervised_loss += equalize_class_loss * class_balance

        return unsupervised_loss, confidence_mask_count