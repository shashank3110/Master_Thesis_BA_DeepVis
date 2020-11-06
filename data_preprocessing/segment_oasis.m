% mri segmentation module

addpath('~/.shashanks/MATLAB/spm12/')
addpath('~/.shashanks/data_preprocessing/')
data_dir='/no_backups/g009/data/OASIS/OASIS1_3_combined'
cd(data_dir);
dir_items = dir();
for i=2167;%i=2053:2199;
   
    % this condition avoids any blank directories
    if startsWith(dir_items(i).name,'.')
    	continue
    end
    img_dir = [data_dir, '/',dir_items(i).name]
	
	img_dir_items=dir(img_dir)
	for j=1:length(img_dir_items);
		if startsWith(img_dir_items(j).name,'.')
    		continue
    	end
		if startsWith(img_dir_items(j).name,'sub-OAS3') | startsWith(img_dir_items(j).name,'OAS1_')
    		nii_file_path(i,1) = {[img_dir, '/',img_dir_items(j).name]}

    		disp([img_dir, '/',img_dir_items(j).name])

    	end
    end

    
end
nii_file_path=nii_file_path(~cellfun('isempty',nii_file_path))
disp(nii_file_path)
spm_segment(nii_file_path)
