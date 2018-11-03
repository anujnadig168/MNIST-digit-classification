function label = extractLabel(file)

i = fopen(file, 'rb');

label = fread(i,inf,'unsigned char');
fclose(i);

label = label(9:end);
end