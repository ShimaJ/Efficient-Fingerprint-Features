%////////////////////feature selection///////////////////////
% %% Example : Relief-F 
% % Parameters
% opts.K  = 3;     % number of nearest neighbors
% opts.Nf = 10;    % select 10 features
% % Load dataset
% % Perform feature selection 
% FS     = jffs('rf',train_data,train_labels,opts);
% % Define index of selected features
% sf_idx = FS.sf;
% % Accuracy  
% kfold  = 5;
% AccFilter    = mSVM(train_data(:,sf_idx),train_labels,kfold);
% %//
% % FS.sf : Index of selected features
% % FS.ff : Selected features
% % FS.nf : Number of selected features
% % FS.s  : Weight or score
% % Acc   : Accuracy of validation model
%/////////////////////////////////////////////////
% % %% Example 1: Particle Swarm Optimization (PSO) 
% % % Number of k in K-nearest neighbor
% % opts.k = 5; 
% % % Ratio of validation data
% % ho = 0.2;
% % % Common parameter settings 
% % opts.N  = 10;     % number of solutions
% % opts.T  = 100;    % maximum number of iterations
% % % Parameters of PSO
% % opts.c1 = 2;
% % opts.c2 = 2;
% % opts.w  = 0.9;
% % HO = cvpartition(train_labels,'HoldOut',ho); 
% % opts.Model = HO; 
% % % Load dataset
% % %//
% % 
% % % Perform feature selection 
% % FS = jfs('pso',train_data,train_labels,opts);%Particle Swarm Optimization;
% % % Define index of selected features
% % sf_idx = FS.sf;
% % % Accuracy  
% % AccWrapper    = jknn(train_data(:,sf_idx),train_labels,opts); 
% % % Plot convergence
% % plot(FS.c); grid on;
% % xlabel('Number of Iterations');
% % ylabel('Fitness Value');
% % title('PSO');

