% ====================================================================
% partB_main.m   |   Μέρος Β – HOG + SVM στο MNIST  (Τελική Έκδοση)
% ====================================================================
%  [1]  HOG (extractHOGFeatures) + SVM για 4 CellSizes
%  [2]  Οπτική παρουσίαση 20 ταξινομήσεων (True vs Pred)
%  [3]  Custom confusion-matrix  + σύγκριση accuracy με CNN (Μέρος Α)
%  [4]  BONUS : Πλήρως χειροποίητο HOG (manual Sobel + histograms)
% --------------------------------------------------------------------
%  Συμβατότητα:  R2015b  →  R2025a  (μόνο char-strings, όλα τα functions
%  κλείνουν με END, καμία εξάρτηση από νεότερη syntax)
% ====================================================================

clear; clc; close all;
fprintf('[Part B]  HOG + SVM on MNIST (MATLAB) – FINAL\n');

%% ------------------------------------------------------------------
% 1.  Φόρτωση ή κατέβασμα του MNIST
% ------------------------------------------------------------------
[XTrain,YTrain,XTest,YTest] = loadMNIST();
YTrain = YTrain(:);          % ως στήλες
YTest  = YTest(:);
imgSize = [28 28];

%% ------------------------------------------------------------------
% 2.  HOG + SVM για διάφορα CellSizes   (Tasks [1] & [2])
% ------------------------------------------------------------------
cellSizeList = { [4 4], [7 7], [8 8], [14 14] };
nCfg   = numel(cellSizeList);
trainAcc = zeros(nCfg,1);
testAcc  = zeros(nCfg,1);

for c = 1:nCfg
    cellSize = cellSizeList{c};
    fprintf('\n[1]  HOG + SVM  (CellSize = [%d %d])\n',cellSize);

    %% 1A.  HOG στο training set ---------------------------------
    tic;
    featRef   = extractHOGFeatures( XTrain(:,:,1), 'CellSize',cellSize );
    hogTrain  = zeros( numel(YTrain), numel(featRef), 'single' );
    for i = 1:numel(YTrain)                       %#ok<PFBNS>
        hogTrain(i,:) = extractHOGFeatures( XTrain(:,:,i), 'CellSize',cellSize );
    end
    fprintf('      HOG(Train)  %d×%d   (%.1fs)\n', ...
            size(hogTrain,1), size(hogTrain,2), toc );

    %% 1B.  SVM (ECOC) -------------------------------------------
    tSVM = templateSVM('KernelFunction','linear','Standardize',true);
    mdl  = fitcecoc( hogTrain, YTrain, 'Learners',tSVM,'Coding','onevsall' );

    %% 1C.  Accuracy στο TRAIN ----------------------------------
    predTrain   = mdl.predict(hogTrain);
    trainAcc(c) = mean(predTrain == YTrain);
    fprintf('      Train Acc : %.2f %%\n', 100*trainAcc(c));

    %% 1D.  HOG + Accuracy στο TEST -----------------------------
    tic;
    hogTest = zeros(numel(YTest), size(hogTrain,2), 'single');
    for i = 1:numel(YTest)                     %#ok<PFBNS>
        hogTest(i,:) = extractHOGFeatures( XTest(:,:,i), 'CellSize',cellSize );
    end
    predTest   = mdl.predict(hogTest);
    testAcc(c) = mean(predTest == YTest);
    fprintf('      Test Acc  : %.2f %%   (HOG(Test) %.1fs)\n', ...
            100*testAcc(c), toc);

    %% 2.  Οπτικά αποτελέσματα  (δείχνουμε μόνο για το 1ο CellSize)  [2]
    if c == 1
        nShow = 20; idx = randperm(numel(YTest), nShow);
        figure('Name',sprintf('Sample classification – CellSize [%d %d]',cellSize));
        for k = 1:nShow
            subplot(4,5,k);
            imshow(XTest(:,:,idx(k)),[]);
            title(sprintf('T:%d | P:%d', YTest(idx(k)), predTest(idx(k))));
        end
        sgtitle(sprintf('Indicative results   (CellSize = [%d %d])',cellSize));
    end

    %% 3.  Confusion Matrix για το τελευταίο CellSize  ----------- [3]
    if c == nCfg
        CM = myConfusionMatrix(YTest, predTest, 10);
        figure('Name','Custom Confusion Matrix');
        imagesc(CM); axis equal tight;
        colormap(parula); colorbar;
        xlabel('Predicted'); ylabel('True');
        title(sprintf('Custom Confusion Matrix – CellSize [%d %d]',cellSize));
        xticks(1:10); yticks(1:10);
        xticklabels(0:9); yticklabels(0:9);
    end
end

%% ------------------------------------------------------------------
% 3.  Σύγκριση accuracy με CNN (Μέρος Α)                         [3]
% ------------------------------------------------------------------
cnnAcc = 0.99;     % ➜ βάλε εδώ το Test-Accuracy του CNN (Μέρος Α)
figure('Name','CNN vs HOG+SVM');
bar([cnnAcc*100, testAcc'.*100]);
set(gca,'XTick',1:5,'XTickLabel',{'CNN','SVM[4 4]','SVM[7 7]','SVM[8 8]','SVM[14 14]'});
ylabel('Accuracy (%)'); ylim([80 100]); grid on;
title('Σύγκριση CNN (Part A) vs SVM/HOG (Part B)');

%% ------------------------------------------------------------------
% 4.  BONUS – Χειροποίητο HOG (χωρίς extractHOGFeatures)          [4]
% ------------------------------------------------------------------
cellSizeB = [4 4];
fprintf('\n[4]  BONUS: manual HOG  (CellSize [%d %d])\n',cellSizeB);

featDimB  = numel( manualHOG( XTrain(:,:,1), cellSizeB ) );
hogTrainB = zeros( numel(YTrain), featDimB, 'single' );
for i = 1:numel(YTrain)                         %#ok<PFBNS>
    hogTrainB(i,:) = manualHOG( XTrain(:,:,i), cellSizeB );
end
tSVM  = templateSVM('KernelFunction','linear','Standardize',true);
mdlB  = fitcecoc( hogTrainB, YTrain, 'Learners',tSVM,'Coding','onevsall' );

hogTestB = zeros( numel(YTest), featDimB, 'single' );
for i = 1:numel(YTest)                          %#ok<PFBNS>
    hogTestB(i,:) = manualHOG( XTest(:,:,i), cellSizeB );
end
accB = mean( mdlB.predict(hogTestB) == YTest );
fprintf('      BONUS Test Accuracy : %.2f %%\n', 100*accB);

%% ==================================================================
% Υποσυναρτήσεις  (όλες κλείνουν με END)
% ==================================================================
function CM = myConfusionMatrix(yTrue, yPred, nClasses)
% myConfusionMatrix : χειροποίητος confusion-matrix  nClasses×nClasses
    CM = zeros(nClasses,nClasses);
    for i = 1:numel(yTrue)
        CM( yTrue(i)+1 , yPred(i)+1 ) = CM( yTrue(i)+1 , yPred(i)+1 ) + 1;
    end
end

% ------------------------------------------------------------------
function feat = manualHOG(I, cellSize)
% manualHOG : HOG χαρακτηριστικά με **χειροποίητα** Sobel & histograms
    if isa(I,'uint8'), I = double(I)/255; end

    % --- Manual Sobel gradients -----------------------------------
    sobX = [-1 0 1; -2 0 2; -1 0 1];
    sobY = [-1 -2 -1; 0 0 0; 1 2 1];
    Gx   = conv2(I, sobX, 'same');
    Gy   = conv2(I, sobY, 'same');
    mag  = hypot(Gx, Gy);
    ori  = atan2d(Gy, Gx);  ori(ori<0) = ori(ori<0) + 180;   % 0–180°

    % --- Histogram per cell ---------------------------------------
    nBins   = 9;  binW = 180/nBins;
    nCellX  = floor(size(I,2) / cellSize(2));
    nCellY  = floor(size(I,1) / cellSize(1));
    feat    = zeros(1, nCellX*nCellY*nBins, 'single');
    idx     = 1;
    for cy = 0:nCellY-1
        for cx = 0:nCellX-1
            rows = cy*cellSize(1)+1 : cy*cellSize(1)+cellSize(1);
            cols = cx*cellSize(2)+1 : cx*cellSize(2)+cellSize(2);
            mCell = mag(rows,cols);
            oCell = ori(rows,cols);
            histo = zeros(1,nBins);
            for b = 0:nBins-1
                mask = oCell >= b*binW & oCell < (b+1)*binW;
                histo(b+1) = sum(mCell(mask));
            end
            feat(idx:idx+nBins-1) = histo;
            idx = idx + nBins;
        end
    end
    feat = feat / (norm(feat) + eps);   % L2-norm
end

% ------------------------------------------------------------------
function [XTrain,YTrain,XTest,YTest] = loadMNIST()
% loadMNIST : κατεβάζει (αν λείπει) & φορτώνει το MNIST, επιστρέφει
%   XTrain  28×28×60000  double  [0,1]
%   YTrain  60000×1      uint8
%   XTest   28×28×10000  double
%   YTest   10000×1      uint8
    baseURL = 'http://yann.lecun.com/exdb/mnist/';
    altURL  = 'https://ossci-datasets.s3.amazonaws.com/mnist/';
    files   = {'train-images-idx3-ubyte','train-labels-idx1-ubyte', ...
               't10k-images-idx3-ubyte','t10k-labels-idx1-ubyte'};
    dataDir = fullfile( tempdir, 'mnist_matlab' );
    if ~exist(char(dataDir),'dir'), mkdir(char(dataDir)); end

    % -- download & unzip ------------------------------------------------
    for k = 1:numel(files)
        fname = files{k};
        raw   = fullfile(dataDir, fname);
        gz    = [raw '.gz'];
        if ~exist(char(raw),'file')
            if ~exist(char(gz),'file')
                fprintf('   ⏬  Downloading %s ...\n', fname);
                try
                    websave(char(gz), [baseURL fname '.gz']);
                catch
                    fprintf('      Primary mirror failed – trying alternative...\n');
                    websave(char(gz), [altURL fname '.gz']);
                end
            end
            gunzip(char(gz), char(dataDir));
        end
    end

    % -- read IDX files --------------------------------------------------
    XTrain = readImgs( fullfile(dataDir,'train-images-idx3-ubyte') );
    YTrain = readLabels( fullfile(dataDir,'train-labels-idx1-ubyte') );
    XTest  = readImgs( fullfile(dataDir,'t10k-images-idx3-ubyte') );
    YTest  = readLabels( fullfile(dataDir,'t10k-labels-idx1-ubyte') );
end

% ------------------------------------------------------------------
function X = readImgs(fname)
% readImgs : διαβάζει αρχείο IDX-3-ubyte (images)
    fid   = fopen(char(fname),'rb');
    fread(fid,1,'uint32','ieee-be');           % magic (discard)
    nImg  = fread(fid,1,'uint32','ieee-be');
    nRow  = fread(fid,1,'uint32','ieee-be');
    nCol  = fread(fid,1,'uint32','ieee-be');
    raw   = fread(fid, nImg*nRow*nCol, 'uint8');
    fclose(fid);
    X = reshape(raw, [nRow,nCol,nImg]);
    X = double(X) / 255;                       % κλίμακα [0,1]
end

% ------------------------------------------------------------------
function y = readLabels(fname)
% readLabels : διαβάζει αρχείο IDX-1-ubyte (labels)
    fid = fopen(char(fname),'rb');
    fread(fid,1,'uint32','ieee-be');           % magic
    n   = fread(fid,1,'uint32','ieee-be');     %#ok<ASGLU>
    y   = fread(fid, n, 'uint8');
    fclose(fid);
end
