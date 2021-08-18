from supervised import Supervised
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import seaborn as sns
from scipy.spatial.distance import cdist # cosine distance
from torch.autograd import grad

class Cgdm(Supervised):
    def __init__(self, encoder, classifier, device):
        super().__init__(encoder, classifier, device)

    def train(generator, classifier_1, classifier_2, source_dataloader, target_dataloader, epochs, hyperparams, save_path):
        # configure hyperparameters
        criterion = nn.CrossEntropyLoss()
        criterion_weighted = weighted_crossentropy

        lr = 3e-4
        num_k = 4
        
        iters = max(len(source_dataloader), len(target_dataloader))

        optimizer_generator = optim.Adam(list(generator.parameters()), lr=lr, weight_decay=5e-4)
        optimizer_classifiers = optim.Adam(list(classifier_1.parameters()) + list(classifier_2.parameters()), lr=lr, weight_decay=5e-4)
            
        scheduler_generator = optim.lr_scheduler.OneCycleLR(optimizer_generator, lr, epochs=epochs, steps_per_epoch=iters)
        scheduler_classifiers = optim.lr_scheduler.OneCycleLR(optimizer_classifiers, lr, epochs=epochs, steps_per_epoch=iters)

        start_epoch = 0

        history = {'epoch_loss_entropy': [], 'epoch_loss_classifier_1': [], 'epoch_loss_classifier_2': [], 'epoch_loss_discrepancy': [], 'accuracy_source': [], 'accuracy_target': []}

        # training loop
        for epoch in range(start_epoch, epochs):
            running_loss_entropy = 0.0
            running_loss_classifier_1 = 0.0
            running_loss_classifier_2 = 0.0
            running_loss_discrepancy = 0.0

            # this is where the unsupervised learning comes in, as such, we're not interested in labels
            for i, ((data_source, labels_source), (data_target, _)) in enumerate(zip(source_dataloader, target_dataloader)):
                if i == 0:
                    pseudo_labels_target = get_pseudo_labels(target_dataloader, generator, classifier_1, classifier_2)

                # set network to training mode
                generator.train()
                classifier_1.train()
                classifier_2.train()

                data_source = data_source.to(device)
                labels_source = labels_source.to(device)

                data_target = data_target.to(device)
                labels_target = pseudo_labels_target[target_dataloader.batch_size*i : target_dataloader.batch_size*(i+1)]
                labels_target = labels_target.to(device)

                # all steps have similar starts
                for phase in [1, 2, 3]:
                    for k in range(num_k): # amount of steps to repeat the generator update
                        # zero gradients
                        optimizer_generator.zero_grad()
                        optimizer_classifiers.zero_grad()
                        
                        # classify the data
                        features_source = generator(data_source)
                        features_target = generator(data_target)

                        outputs_1_source = classifier_1(features_source)
                        outputs_1_target = classifier_1(features_target)
                        outputs_2_source = classifier_2(features_source)
                        outputs_2_target = classifier_2(features_target)

                        # get losses
                        entropy_loss = entropy(outputs_1_target) + entropy(outputs_2_target)

                        loss_1 = criterion(outputs_1_source, labels_source)
                        loss_2 = criterion(outputs_2_source, labels_source)

                        if phase == 1:
                            # train networks to minimize loss on source
                            supervised_loss = criterion_weighted(outputs_1_target, labels_target) + criterion_weighted(outputs_2_target, labels_target)

                            loss = loss_1 + loss_2 + (0.01 * entropy_loss) + (0.01 * supervised_loss)

                            # backpropagate and update weights
                            loss.backward()
                            #optimizer_generator.step()
                            #ptimizer_classifiers.step()
                            xm.optimizer_step(optimizer_generator, barrier=True)
                            xm.optimizer_step(optimizer_classifiers, barrier=True)

                            # exit generator loop (num_k)
                            break

                        elif phase == 2:
                            # train classifiers to maximize divergence between classifier outputs on target (without labels)
                            discrepancy_loss = discrepancy(outputs_1_target, outputs_2_target)
                            loss = loss_1 + loss_2 - (1.0 * discrepancy_loss) + (0.01 * entropy_loss) 
                            
                            # backpropagate and update weights
                            loss.backward()
                            #optimizer_classifiers.step()
                            xm.optimizer_step(optimizer_classifiers, barrier=True)

                            # exit generator loop (num_k)
                            break

                        elif phase == 3:
                            # train generator to minimize divergence between classifier outputs with gradient similarity
                            discrepancy_loss = discrepancy(outputs_1_target, outputs_2_target)

                            source_pack = (outputs_1_source, outputs_2_source, labels_source)
                            target_pack = (outputs_1_target, outputs_2_target, labels_target)

                            gradient_discrepancy_loss = gradient_discrepancy(source_pack, target_pack, generator, classifier_1, classifier_2)
                            loss = (1.0 * discrepancy_loss) + (0.01 * entropy_loss) + (0.01 * gradient_discrepancy_loss)

                            # backpropagate and update weights
                            loss.backward()
                            #optimizer_generator.step()
                            xm.optimizer_step(optimizer_generator, barrier=True)

                # metrics
                running_loss_entropy += entropy_loss.item()
                running_loss_classifier_1 += loss_1.item()
                running_loss_classifier_2 += loss_2.item()
                running_loss_discrepancy += discrepancy_loss.item()
                
                # scheduler
                scheduler_generator.step()
                scheduler_classifiers.step()

            # get losses
            # we use np.min because zip only goes up to the smallest list length
            epoch_loss_entropy = running_loss_entropy / iters
            epoch_loss_classifier_1 = running_loss_classifier_1 / iters
            epoch_loss_classifier_2 = running_loss_classifier_2 / iters
            epoch_loss_discrepancy = running_loss_discrepancy / iters

            history['epoch_loss_entropy'].append(epoch_loss_entropy)
            history['epoch_loss_classifier_1'].append(epoch_loss_classifier_1)
            history['epoch_loss_classifier_2'].append(epoch_loss_classifier_2)
            history['epoch_loss_discrepancy'].append(epoch_loss_discrepancy)

            # evaluate on training data
            epoch_accuracy_source, epoch_auc_source = evaluate(generator, classifier_1, classifier_2, source_dataloader)
            epoch_accuracy_target, epoch_auc_target = evaluate(generator, classifier_1, classifier_2, target_dataloader)
            history['accuracy_source'].append(epoch_accuracy_source)
            history['accuracy_target'].append(epoch_accuracy_target)

            # save checkpoint
            torch.save({'generator_weights': generator.state_dict(),
                        'classifier_1_weights': classifier_1.state_dict(),
                        'classifier_2_weights': classifier_2.state_dict(),
                        'epoch': epoch,
                        'history': history}, save_path)

            print('[Epoch {}/{}] entropy loss: {:.6f}; classifier 1 loss: {:.6f}; classifier 2 loss: {:.6f}; discrepancy loss: {:.6f}'.format(epoch+1, epochs, epoch_loss_entropy, epoch_loss_classifier_1, epoch_loss_classifier_2, epoch_loss_discrepancy))
            print('[Epoch {}/{}] accuracy source: {:.6f}; auc score source: {:.6f}; accuracy target: {:.6f}; auc score target: {:.6f}'.format(epoch+1, epochs, epoch_accuracy_source, epoch_auc_source, epoch_accuracy_target, epoch_auc_target))

        return generator, classifier_1, classifier_2, history

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
    def _weighted_crossentropy(input, target):
        input_softmax = F.softmax(input, dim=1)
        entropy = -input_softmax * torch.log(input_softmax + 1e-5) # standard info entropy with "anti-zero" term
        entropy = torch.sum(entropy, dim=1)

        weight = 1.0 + torch.exp(-entropy)
        weight = weight / torch.sum(weight).detach().item()

        return torch.mean(weight * F.cross_entropy(input, target, reduction='none'))

    @staticmethod
    def _entropy(input, epsilon=1e-5):
        # apply softmax
        input = F.softmax(input, dim=1)
        
        # entropy_condition
        entropy_condition = -input * torch.log(input + epsilon)
        entropy_condition = torch.sum(entropy_condition, dim=1).mean()
        
        # entropy_div
        input = torch.mean(input, 0) + epsilon
        entropy_div = input * torch.log(input)
        entropy_div = torch.sum(entropy_div)

        return entropy_condition + entropy_div

    @staticmethod
    def _discrepancy(input_1, input_2):
        return torch.mean(torch.abs(F.softmax(input_1, dim=1) - F.softmax(input_2, dim=1)))

    @staticmethod
    def _gradient_discrepancy(source_pack, target_pack, generator, classifier_1, classifier_2):
        outputs_1_source, outputs_2_source, labels_source = source_pack
        outputs_1_target, outputs_2_target, labels_target = target_pack

        criterion = nn.CrossEntropyLoss()
        criterion_weighted = weighted_crossentropy

        gradient_loss = 0

        # get losses
        loss_1_source = criterion(outputs_1_source, labels_source)
        loss_2_source = criterion(outputs_2_source, labels_source)
        losses_source = [loss_1_source, loss_2_source]

        loss_1_target = criterion_weighted(outputs_1_target, labels_target)
        loss_2_target = criterion_weighted(outputs_2_target, labels_target)
        losses_target = [loss_1_target, loss_2_target]

        # get gradient loss from each classifier
        for classifier, loss_source, loss_target in zip([classifier_1, classifier_2], losses_source, losses_target):
            grad_cosine_similarity = []
            
            for name, params in classifier.named_parameters():
                real_grad = grad([loss_source], [params], create_graph=True, only_inputs=True, allow_unused=False)[0]
                fake_grad = grad([loss_target], [params], create_graph=True, only_inputs=True, allow_unused=False)[0]

                if len(params.shape) > 1:
                    cosine_similarity = F.cosine_similarity(fake_grad, real_grad, dim=1).mean()
                else:
                    cosine_similarity = F.cosine_similarity(fake_grad, real_grad, dim=0)

                grad_cosine_similarity.append(cosine_similarity)

            # concatenate cosine similarities
            grad_cosine_similarity = torch.stack(grad_cosine_similarity)

            # get loss for this classifier
            gradient_loss += (1.0 - grad_cosine_similarity).mean()

        return gradient_loss/2.0 # mean of both gradient_loss(es)

    @staticmethod
    def _get_pseudo_labels(target_dataloader, generator, classifier_1, classifier_2):
        generator.eval()
        classifier_1.eval()
        classifier_2.eval()

        start_test = True

        with torch.no_grad():
            for data, labels in target_dataloader:
                data = data.to(device)
                labels = labels.to(device)

                # generate features
                features = generator(data)

                outputs_1 = classifier_1(features)
                outputs_2 = classifier_2(features)
                outputs = outputs_1 + outputs_2

                if start_test:
                    all_features = features.float().cpu()
                    all_outputs = outputs.float().cpu()
                    all_labels = labels.float().cpu()
                    start_test = False
                
                else:
                    all_features = torch.cat((all_features, features.float().cpu()), dim=0)
                    all_outputs = torch.cat((all_outputs, outputs.float().cpu()), dim=0)
                    all_labels = torch.cat((all_labels, labels.float().cpu()), dim=0)

        all_outputs = F.softmax(all_outputs, dim=1)
        _, preds = torch.max(all_outputs, dim=1)
        accuracy = torch.sum(torch.squeeze(preds).float() == all_labels).item() / float(all_labels.size()[0])

        all_features = torch.cat((all_features, torch.ones(all_features.size(0), 1)), dim=1)
        all_features = (all_features.t() / torch.norm(all_features, p=2, dim=1)).t()
        all_features = all_features.float().cpu().numpy()

        # perform k-means clustering
        k = all_outputs.size(1)

        for i in range(2):
            if i == 0:
                aff = all_outputs.float().cpu().numpy()
            else:
                aff = np.eye(k)[preds_label]
            
            initial_centroid = aff.transpose().dot(all_features)
            initial_centroid = initial_centroid / (1e-8 + aff.sum(axis=0)[:,None])
            distance = cdist(all_features, initial_centroid, 'cosine')

            preds_label = distance.argmin(axis=1)
            accuracy_kmeans = np.sum(preds_label == all_labels.float().cpu().numpy()) / len(all_features)

        print('Only source accuracy = {:.2f}% -> After the clustering = {:.2f}%'.format(accuracy*100, accuracy_kmeans*100))
        return torch.tensor(preds_label, dtype=torch.long)