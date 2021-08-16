from supervised import Supervised
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

class Adamatch(Supervised):
    def __init__(self, encoder, classifier, device):
        super().__init__(encoder, classifier, device)

    def train_target(self, source_dataloader_weak, source_dataloader_strong,
                     target_dataloader_weak, target_dataloader_strong, target_dataloader_test,
                     epochs, hyperparams, save_path):
        # configure hyperparameters
        lr = hyperparams['learning_rate']
        wd = hyperparams['weight_decay']
        cyclic_scheduler = hyperparams['cyclic_scheduler']
        tau = 0.9
        
        iters = max(len(source_dataloader_weak), len(source_dataloader_strong), len(target_dataloader_weak), len(target_dataloader_strong))

        # mu related stuff
        steps_per_epoch = iters
        total_steps = epochs * steps_per_epoch 
        current_step = 0

        # configure optimizer and scheduler
        optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.classifier.parameters()), lr=lr, weight_decay=wd)
        if cyclic_scheduler:
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=iters)

        # early stopping variables
        start_epoch = 0
        best_acc = 0.0
        patience = hyperparams['patience']
        bad_epochs = 0

        self.history = {'epoch_loss': [], 'accuracy_source': [], 'accuracy_target': []}

        # training loop
        for epoch in range(start_epoch, epochs):
            running_loss = 0.0

            # set network to training mode
            self.encoder.train()
            self.classifier.train()

            dataset = zip(source_dataloader_weak, source_dataloader_strong, target_dataloader_weak, target_dataloader_strong)

            # this is where the unsupervised learning comes in, as such, we're not interested in labels
            for (data_source_weak, labels_source), (data_source_strong, _), (data_target_weak, _), (data_target_strong, _) in dataset:
                data_source_weak = data_source_weak.to(self.device)
                labels_source = labels_source.to(self.device)

                data_source_strong = data_source_strong.to(self.device)
                data_target_weak = data_target_weak.to(self.device)
                data_target_strong = data_target_strong.to(self.device)

                # concatenate data (in case of low GPU power this could be done after classifying with the model)
                data_combined = torch.cat([data_source_weak, data_source_strong, data_target_weak, data_target_strong], 0)
                source_combined = torch.cat([data_source_weak, data_source_strong], 0)

                # get source data limit (useful for slicing later)
                source_total = source_combined.size(0)

                # zero gradients
                optimizer.zero_grad()

                # forward pass: calls the model once for both source and target and once for source only
                logits_combined = self.classifier(self.encoder(data_combined))
                logits_source_p = logits_combined[:source_total]

                # from https://github.com/yizhe-ang/AdaMatch-PyTorch/blob/main/trainers/adamatch.py
                self._disable_batchnorm_tracking(self.encoder)
                self._disable_batchnorm_tracking(self.classifier)
                logits_source_pp = self.classifier(self.encoder(source_combined))
                
                self._enable_batchnorm_tracking(self.encoder)
                self._enable_batchnorm_tracking(self.classifier)

                # perform random logit interpolation
                lambd = torch.rand_like(logits_source_p).to(self.device)
                final_logits_source = (lambd * logits_source_p) + ((1-lambd) * logits_source_pp)

                # distribution allignment
                ## softmax for logits of weakly augmented source images
                logits_source_weak = final_logits_source[:data_source_weak.size(0)]
                pseudolabels_source = F.softmax(logits_source_weak, 1)

                ## softmax for logits of weakly augmented target images
                logits_target = logits_combined[source_total:]
                logits_target_weak = logits_target[:data_target_weak.size(0)]
                pseudolabels_target = F.softmax(logits_target_weak, 1)

                ## allign target label distribtion to source label distribution
                expectation_ratio = (1e-6 + torch.mean(pseudolabels_source)) / (1e-6 + torch.mean(pseudolabels_target))
                final_pseudolabels = F.normalize((pseudolabels_target * expectation_ratio), p=2, dim=1) # L2 normalization

                # perform relative confidence thresholding
                row_wise_max, _ = torch.max(pseudolabels_source, dim=1)
                final_sum = torch.mean(row_wise_max, 0)
                
                ## define relative confidence threshold
                c_tau = tau * final_sum

                max_values, _ = torch.max(final_pseudolabels, dim=1)
                mask = (max_values >= c_tau).float()

                # compute loss
                source_loss = self._compute_source_loss(logits_source_weak, final_logits_source[data_source_weak.size(0):], labels_source)
                
                final_pseudolabels = torch.max(final_pseudolabels, 1)[1] # argmax
                target_loss = self._compute_target_loss(final_pseudolabels, logits_target[data_target_weak.size(0):], mask)

                ## compute target loss weight (mu)
                pi = torch.tensor(np.pi, dtype=torch.float).to(self.device)
                step = torch.tensor(current_step, dtype=torch.float).to(self.device)
                mu = 0.5 - torch.cos(torch.minimum(pi, (2*pi*step) / total_steps)) / 2

                ## get total loss
                loss = source_loss + (mu * target_loss)
                current_step += 1

                # backpropagate and update weights
                loss.backward()
                optimizer.step()
                if cyclic_scheduler:
                    scheduler.step()

                # metrics
                running_loss += loss.item()

            # get losses
            # we use np.min because zip only goes up to the smallest list length
            epoch_loss = running_loss / iters
            self.history['epoch_loss'].append(epoch_loss)

            # self.evaluate on testing data (target domain)
            epoch_accuracy_source = self.evaluate(self.encoder, self.classifier, source_dataloader_weak)
            epoch_accuracy_target = self.evaluate(self.encoder, self.classifier, target_dataloader_weak)
            test_epoch_accuracy = self.evaluate(self.encoder, self.classifier, target_dataloader_test)
            
            self.history['accuracy_source'].append(epoch_accuracy_source)
            self.history['accuracy_target'].append(epoch_accuracy_target)

            # save checkpoint
            if test_epoch_accuracy > best_acc:
                torch.save({'encoder_weights': self.encoder.state_dict(),
                            'classifier_weights': self.classifier.state_dict()
                        }, save_path)
                best_acc = test_epoch_accuracy
                bad_epochs = 0
                
            else:
                bad_epochs += 1
                
            print('[Epoch {}/{}] loss: {:.6f}; accuracy source: {:.6f}; accuracy target: {:.6f}; val accuracy: {:.6f};'.format(epoch+1, epochs, epoch_loss, epoch_accuracy_source, epoch_accuracy_target, test_epoch_accuracy))
            
            if bad_epochs >= patience:
                print(f"reached {bad_epochs} bad epochs, stopping training with best val accuracy of {best_acc}!")
                break

        best = torch.load(save_path)
        self.encoder.load_state_dict(best['encoder_weights'])
        self.classifier.load_state_dict(best['classifier_weights'])
        
        return self.encoder, self.classifier, self.history

    def plot_metrics(self):
        # plot metrics for losses n stuff
        fig, axs = plt.subplots(1, 3, figsize=(18,5), dpi=200)

        epochs = len(self.history['epoch_loss'])

        axs[0].plot(range(1, epochs+1), self.history['epoch_loss'])
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Entropy loss')

        axs[1].plot(range(1, epochs+1), self.history['accuracy_source'])
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy')
        axs[1].set_title('Accuracy on weakly augmented source')

        axs[2].plot(range(1, epochs+1), self.history['accuracy_target'])
        axs[2].set_xlabel('Epochs')
        axs[2].set_ylabel('Accuracy')
        axs[2].set_title('Accuracy on weakly augmented target')      
            
        plt.show()

    @staticmethod
    def _disable_batchnorm_tracking(model):
        def fn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.track_running_stats = False

        model.apply(fn)

    @staticmethod
    def _enable_batchnorm_tracking(model):
        def fn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.track_running_stats = True

        model.apply(fn)

    @staticmethod
    def _compute_source_loss(logits_weak, logits_strong, labels):
        """
        Receives logits as input (dense layer outputs with no activation function)
        """
        loss_function = nn.CrossEntropyLoss() # default: `reduction="mean"`
        weak_loss = loss_function(logits_weak, labels)
        strong_loss = loss_function(logits_strong, labels)

        #return weak_loss + strong_loss
        return (weak_loss + strong_loss) / 2

    @staticmethod
    def _compute_target_loss(pseudolabels, logits_strong, mask):
        """
        Receives logits as input (dense layer outputs with no activation function).
        `pseudolabels` are treated as ground truth (standard SSL practice).
        """
        loss_function = nn.CrossEntropyLoss(reduction="none")
        pseudolabels = pseudolabels.detach() # remove from backpropagation

        loss = loss_function(logits_strong, pseudolabels)
        
        return (loss * mask).mean()