function image = extractImage(file)

i = fopen(file, 'rb');

[~] = fread(i,1,'int32',0,'b');
iminfo=fread(i,3,'int32',0,'b');

image = fread(i,inf,'unsigned char');
fclose(i);

image = reshape(image,iminfo(2),iminfo(3),iminfo(1));
image = double(image)/255;
end
