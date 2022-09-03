# /bin/zsh

file_prefix=$1;
file_extentions=(
	'aux'
	'log'
	'nav'
	'out'
	'snm'
	'toc'
)

for file_extention in ${file_extentions[@]}; do
	echo 'deleting the file' $file_prefix.$file_extention;
	rm $file_prefix.$file_extention;
done
