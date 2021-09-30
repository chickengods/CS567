%% What is MATLAB capable of displaying?

% Goal:  Explore display range of MATLAB

% On your own.  Plot the vector -10:70 using image() and grayscale.  
% What do you notice? 64 vs 256

figure
dat = rand(10,10)*1000;
image(dat)
colormap(gray)
figure
imagesc(dat)
colormap(gray)

%figure
%image([-10:70])
%colormap(gray)

%figure
%imagesc([-10:70])
%colormap(gray)




%% Reading in a Dicom
% This is from OpenDicom_3.m (textbook files)

clear
path = 'C:\Users\cs_vi\Dropbox\BMI567-2021\Data\LessonData\3_ImageRepresentation';
fpointer=fopen(fullfile(path,'PIG_MR'),'r');
f = dir(fullfile(path,'PIG_MR'));
f.bytes
% Note the header size will vary by dcm, so you need some extra info.   In
% this case we know the image size is 512x512 and is 
%  Basically, you'll need an extra DICOM-specific program to figure
%  it out.

% Somebody tells us the image is 512 x 512 and is 16 bit = 2bytes, so we
% can compute the header sizec
hdr_size = f.bytes - 512*512*2;

% What if somebody doesn't tell you?  Check out dicominfo()
dicominfo(fullfile(path,'PIG_MR'))

% What does fseek do?  (look it up now)
fseek(fpointer,hdr_size,'bof')

%fseek(fpointer,10000,'bof')

img=zeros(512,512);
% short = 16 bits = 2 bytes.  More listed here: https://www.mathworks.com/help/matlab/ref/fread.html
img(:)=fread(fpointer,(512*512),'short');
img=transpose(img);

figure
colormap(gray)
image(img)

figure
imagesc(img)
colormap(gray)

figure
hist(img(:), 100)

figure
imagesc(img>220)

%figure
%hist(img)

% close it up, since we got what we want
fclose(fpointer);

% Let's repeat the above, but give it the wrong starting point



% Also, you can just use dicomread (much faster)
img2 = double(dicomread(fullfile(path,'PIG_MR')));
figure
subplot(1,2,1)
imagesc(img2)
colormap(gray)
colorbar
subplot(1,2,2)
hist(img2(:), 100)
xticks(0:10:63)

%% Conversion to 6 bit integer: ch 4 ERROR IN BOOK!!!  Look carefully!
% This is based on Practical Lesson 4.5.1

% Although we will eventually use imagesc for scaling, we must learn how to
% scale first

%  What range do we want our image values to be to display nicely in
%  MATLAB?

% What is the current range of values in img?

% General instructions:
% Step 1:  Ensure smallest value is 0 
% It already is in our image, but this code will work on images that
%    don't start at 0.

imin=min(img(:));
% Subtracting the minimum will set the new min to 0
img_scale=img-imin;
% Next step is to find the max.
% Divide by old max and multiply by what we'd like the new max to be and
% take floor to convert to integer again
imax= max(img_scale(:));  
img_scale = round(img_scale/imax*255);


figure
image(img_scale)
colormap(gray)

%% Sigmoid function

img = 0:63;

w = 10;
s = 2;
s_w_10_s_2 = 63./(1+exp(-1*(img - w)/s));

figure
plot(img, s_w_10_s_2)


w = 10;
s = 5;
s_w_10_s_5 = 63./(1+exp(-1*(img - w)/s));

figure
subplot(2, 1, 1)
plot(img, s_w_10_s_2)
subplot(2,1,2)
plot(img, s_w_10_s_5)





%% Making a histogram without using hist()
% From Birkfellner's Histogram_4.m file, but his code won't generalize well
% Please use this code instead of what is in Histogram_4.m
clear
path = 'C:\Users\cs_vi\Dropbox\BMI567-2021\Data\LessonData\4_ImageIntensity';    ;
% Please change to double when you read in data!
%  The author doesn't always do this
img = double(imread(fullfile(path,'ABD_CT.jpg')));
depth = max(max(img))-min(min(img))
num_bins=16;  %number of bins for histogram 
min_img = min(img(:));
% First check that the range is right
img2 = ceil((img-min_img)*(num_bins)/depth);
min(img2(:))
max(img2(:))
% Only wrong at 0
img2(img2==0) = 1;
%  note, img2 only has the bin assignments in a matrix, it isn't
%  the histogram.  The following loop can do that.
figure
colormap(gray)
imagesc(img2)
% This is the loop the author used, but you
% can do this more efficiently
hist16 = zeros(num_bins,1);
for i = 1:261
  for j = 1:435
    rho = img(i,j);
    % I had to edit the next line!
    b16 = ceil(num_bins*(rho-min_img)/depth);
    % Need to catch the case where b16 is 0
    if (b16 == 0)
        b16 = 1;
    end
    hist16(b16,1)=hist16(b16,1)+1;
  end
end
bar(hist16)

% Let's compare to hist
subplot(1, 2, 1)
bar(hist16)
subplot(1, 2, 2)
hist(img(:), num_bins)

% Matches!
[hist16'; hist(img(:), num_bins)]

