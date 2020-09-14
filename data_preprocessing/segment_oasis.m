% List of open inputs
% nii_file = 'S:\Auswertung\Sergios\Sotirios\Normkollektiv\sNR_NORM_052-0003-00001-000192-01.nii';
%nii_file = 'S:\Auswertung\Sergios\Sotirios\Normkollektiv\T1.nii';
%data_dir = '/media/shashanks/Windows/Users/Shashank_S/linux_partition/BA_estimation/OASIS2/OAS2_RAW_PART1';
addpath('~/.shashanks/MATLAB/spm12/')
addpath('~/.shashanks/data_preprocessing/')
data_dir='/no_backups/g009/data/OASIS/OASIS1_3_combined'
cd(data_dir);
dir_items = dir();
for i=2167;%i=2053:2199;
    % disp([data_dir, '/',dir_items(i).name,'/RAW/','mpr-1.nifti.img'])
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
% nii_file_path = {[data_dir, '/',dir_items(1).name,'/RAW/','mpr-1.nifti.nii']} 
% spm_segment(nii_file_path)