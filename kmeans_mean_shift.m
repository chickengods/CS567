%% Kmeans
%%
file_path = 'C:\Users\cs_vi\Dropbox\BMI567-2021\week9';

% data (slice of MRI data: T1 structural image of brain)
load(sprintf('%s/T1.mat', file_path));

figure
colormap(gray)
imagesc(T1)

% First put data in a vector and only include nonzero values
T1_nonzero = T1(T1>0);
size(T1_nonzero)

num_vox = length(T1_nonzero);

% Have a look at the data and histogram
figure
subplot(2, 1, 1)
imagesc(T1); colormap('gray')
subplot(2,1,2)
hist(T1_nonzero, 60)

figure
colormap(gray)
imagesc(T1>0)



% Step 1: Initialize clusters
% Randomly sample values from [1, 2, 3] with replacement
K = 3;
current_cluster = randsample(K, num_vox, true);

%initialize for while loop
change = ones(size(current_cluster));
count = 0;
while sum(abs(change)) ~= 0 % Keep going until the cluster assignment
                      % doesn't change
    count = count + 1
    % 2a. Compute cluster mean (centroid)
    mn_vec = zeros(1, K);
    for k=1:K
        mn_vec(k) = mean(T1_nonzero(current_cluster == k));
    end
    
    % 2b. Reassign according to closest mean (how could you speed this up?)
    new_cluster = zeros(num_vox, 1);
    for vox_ind = 1:num_vox
        dist_means = (T1_nonzero(vox_ind)*ones(1, K) - mn_vec).^2;
        new_cluster(vox_ind) = find(dist_means == min(dist_means));
    end
    change = new_cluster - current_cluster;
    current_cluster = new_cluster;
    unique(current_cluster)
end
count

figure
seg_img = 0*T1;
seg_img(T1>0)=current_cluster;
subplot(3, 1,1)
imagesc(seg_img == 1);colormap('gray')
subplot(3, 1, 2)
imagesc(seg_img == 2)
subplot(3, 1, 3)
imagesc(seg_img == 3)



%%
% To get the best solution, run many times and minimize the cost
num_iter = 10;
cost_vec = ones(1, num_iter);
cluster_solution = zeros(num_vox, num_iter);

for iter = 1:num_iter
  iter
  current_cluster = randsample(K, num_vox, true);
%initialize for while loop
  change = ones(size(current_cluster));
  count = 0;
  while sum(abs(change)) ~= 0 % Keep going until the cluster assignment
                      % doesn't change
    count = count + 1;
    % 2a. Compute cluster mean (centroid)
    mn_vec = zeros(1, K);
    for k=1:K
        mn_vec(k) = mean(T1_nonzero(current_cluster == k));
    end
    
    % 2b. Reassign according to closest mean (how could you speed this up?)
    new_cluster = zeros(num_vox, 1);
    for vox_ind = 1:num_vox
        dist_means = sum((T1_nonzero(vox_ind)*ones(1, K) - mn_vec).^2, ...
                         1);
        new_cluster(vox_ind) = find(dist_means == min(dist_means));
    end
    change = new_cluster - current_cluster;
    current_cluster = new_cluster;
    
    sum_clust_dist = zeros(1, K);
    for k=1:K
        dat_current_cluster = T1_nonzero(current_cluster == k);
        sum_clust_dist(k) = sum((dat_current_cluster - ...
                                 mean(dat_current_cluster)).^2);
    end
    cur_cost = sum(sum_clust_dist);
    
  end
  cluster_solution(:,iter) = current_cluster;
  cost_vec(iter) = sum(sum_clust_dist);
end

min_loc = find(cost_vec==min(cost_vec));

final_cluster = cluster_solution(:,min_loc(1));

figure
seg_img = zeros(size(T1));
seg_img(T1>0)= final_cluster;
subplot(2, 2, 1)
imagesc(T1); colormap('gray')
subplot(2, 2,2)
imagesc(seg_img == 1);colormap('gray')
subplot(2, 2, 3)
imagesc(seg_img == 2)
subplot(2, 2, 4)
imagesc(seg_img == 3)

% Have a look at the kmeans function

idx = kmeans(T1_nonzero, 3);

figure
seg_img = T1;
seg_img(T1>0)= idx;
subplot(2, 2, 1)
imagesc(T1); colormap('gray')
subplot(2, 2,2)
imagesc(seg_img == 1);colormap('gray')
subplot(2, 2, 3)
imagesc(seg_img == 2)
subplot(2, 2, 4)
imagesc(seg_img == 3)

idx4 = kmeans(T1_nonzero, 4);
seg_img = T1;
seg_img(T1>0)=idx4;
figure
imagesc(seg_img)

% Let's view the thresholds on the histogram

% What would the thresholds be??
figure
subplot(3,1,1)
hist(T1_nonzero(final_cluster == 1))
xlim([min(T1_nonzero), max(T1_nonzero)])
subplot(3, 1, 2)
hist(T1_nonzero(final_cluster == 2))
xlim([min(T1_nonzero), max(T1_nonzero)])
subplot(3,1,3)
hist(T1_nonzero(final_cluster == 3))
xlim([min(T1_nonzero), max(T1_nonzero)])

min(T1_nonzero(final_cluster==1))
max(T1_nonzero(final_cluster==1))

min(T1_nonzero(final_cluster==2))
max(T1_nonzero(final_cluster==2))

min(T1_nonzero(final_cluster==3))
max(T1_nonzero(final_cluster==3))

thresh1 = 45.5;
thresh2 = 85.5;

figure
subplot(2, 1, 1)
imagesc(T1); colormap('gray')
subplot(2,1,2)
hist(T1_nonzero, 20)
hold on
line([thresh1, thresh1], ylim, 'LineWidth', 2)
line([thresh2, thresh2], ylim, 'linewidth', 2)
hold off

%%
%%  Mean shift
%%

% Quick 1 feature example
dat = [randn(500, 1)*3; randn(500, 1)*4+15];

% Get histogram of density in Matlab
hist(dat, 40)
[f, x] = hist(dat, 40);
dx = x(2)-x(1);

figure
bar(x, f/sum(f.*dx)); 

%estimate kde in 1D using Epanechnikov kernel

x_val = min(dat):.1:max(dat);
dens_est = zeros(size(x_val));
h=.5;
nx = length(dat);

for ind=1:length(x_val)
   val_loop = dat(abs((dat-x_val(ind))/h) <=1); 
   ku = .75*(1-((x_val(ind) - val_loop)/h).^2);
   dens_est(ind) = 1/(nx*h)*sum(ku);
end

figure
bar(x, f/sum(f.*dx)); 
hold on
plot(x_val, dens_est, 'r', 'linewidth', 3)
hold off



% Today we will use 2 features:  A T1 structural MRI and a T2 structural
% MRI

% Image segmentation using k-means
file_path = 'C:\Users\cs_vi\Dropbox\BMI567-2021\week9';

% data (slice of MRI data: T1 structural image of brain)
load(sprintf('%s/T1.mat', file_path));

% Same, but a T2 image
load(sprintf('%s/T2.mat', file_path));

% The two images
figure
subplot(1, 2, 1)
imagesc(T1); colormap('gray')
subplot(1,2,2)
imagesc(T2); colormap('gray')


% To save time, just using a small section of data for illustration
T1_section = double(T1(80:100, 100:150));
T2_section = double(T2(80:100, 100:150));

figure
subplot(1, 2, 1)
imagesc(T1_section); colormap('gray')
subplot(1,2,2)
imagesc(T2_section); colormap('gray')

% plot the intensity values against each other
figure
plot(T1_section, T2_section, 'b.')
xlabel('T1 image', 'Fontsize', 20)
ylabel('T2 image', 'Fontsize', 20)


% put data in vector and remove 0 (0 is background)
keep = T1_section>0 & T2_section>0;
T1_vec = T1_section(keep);
T2_vec = T2_section(keep);

% rescale
T1_vec = (T1_vec-mean(T1_vec))/std(T1_vec);
T2_vec = (T2_vec-mean(T2_vec))/std(T2_vec);

figure
plot(T1_vec, T2_vec, 'b.')

% Run mean shift and plot shifts as we go
radius = 1;
stop_criterion = .05;  % Largest distance allowed between 2
                       % iterations without stopping

seg_feat1 = 0*T1_vec;
seg_feat2 = 0*T2_vec;

figure
plot(T1_vec, T2_vec, 'black.', 'markersize', 20)
hold on

% Loop through each voxel
for loc=1:length(T1_vec)
    cur_feat1 = T1_vec(loc);
    cur_feat2 = T2_vec(loc);
    cur_criterion = 10;
    % Find the convergence point for the single voxel
    while cur_criterion>stop_criterion
    % compute distance for all points vs first
        dist_loop = ((cur_feat1(end) - T1_vec).^2 + (cur_feat2(end) - T2_vec).^2).^.5;
        keep_loop = dist_loop<radius;
        mean_loop_feat1 = mean(T1_vec(keep_loop==1));
        mean_loop_feat2 = mean(T2_vec(keep_loop==1));
        cur_criterion = ((cur_feat1(end) - mean_loop_feat1).^2 + (cur_feat2(end) - mean_loop_feat2).^2).^.5;
        cur_feat1 = [cur_feat1, mean_loop_feat1];
        cur_feat2 = [cur_feat2, mean_loop_feat2];
    end
    plot(cur_feat1, cur_feat2, 'green-')
    seg_feat1(loc) = cur_feat1(end);
    seg_feat2(loc) = cur_feat2(end);  
end


plot(seg_feat1, seg_feat2, 'bluex','markersize', 20, 'linewidth', 3)
hold off

% Round and look at uniqe end points
unique_vals = unique([round(seg_feat1), round(seg_feat2)], 'rows')

size(unique([(seg_feat1), (seg_feat2)], 'rows'))


% Let's create the 3 classes and plot
seg_vec = 0*T1_vec;
seg_vec(round(seg_feat1)==-2 & round(seg_feat2)== -3) = 1;
seg_vec(round(seg_feat1)==-2 & round(seg_feat2)== -2) = 1;
seg_vec(round(seg_feat1)==-2 & round(seg_feat2)== 3) = 2;
seg_vec(round(seg_feat1)==0 & round(seg_feat2)== 0) = 3;
seg_vec(round(seg_feat1)==1 & round(seg_feat2)== 0) = 4;


seg_img = T1_section;
seg_img(keep==1) = seg_vec;

figure
subplot(1, 2, 1)
imagesc(T1_section); colormap('gray')
subplot(1,2,2)
imagesc(T2_section); colormap('gray')

figure
subplot(2,2,1)
imagesc(seg_img==1); colormap('gray')
subplot(2,2,2)
imagesc(seg_img==2); colormap('gray')
subplot(2,2,3)
imagesc(seg_img==3); colormap('gray')
subplot(2,2,4)
imagesc(seg_img==4); colormap('gray')



