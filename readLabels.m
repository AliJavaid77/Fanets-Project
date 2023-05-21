%function labels = readLabels(filename)
%    labelImage = imread(filename);
%    labelImage = imresize(labelImage, [224 224]);
%    labelImage = imbinarize(labelImage); % Binarize the mask image
%    labels = categorical(labelImage, [0, 1], {'background', 'instrument'});
%end
function labels = readLabels(filename)
    labelImage = imread(filename);
    labelImage = imresize(labelImage, [224 224]);
    labelImage = imbinarize(labelImage); % Binarize the mask image
    labels = labelImage;
end

