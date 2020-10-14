

addpath('/no_backups/g009/data/OASIS/')
addpath('~/.shashanks/MATLAB/spm12')
data_dir='/no_backups/g009/data/OASIS/OASIS1_3_combined'
cd(data_dir);
dir_items = dir();
clear jobs;

k=1
for i=1:length(dir_items);

if startsWith(dir_items(i).name,'.')
    	continue
    end
    img_dir = [data_dir, '/',dir_items(i).name]
	
	img_dir_items=dir(img_dir)
	for j=1:length(img_dir_items);
		if startsWith(img_dir_items(j).name,'.')
    		continue
    	end
		if startsWith(img_dir_items(j).name,'rc1') 
    		%nii_file_path(i,1) = {[img_dir, '/',img_dir_items(j).name]}
    		c1f(k,1) = {[img_dir, '/',img_dir_items(j).name,',1']}
                end
                if startsWith(img_dir_items(j).name,'rc2')
    		c2f(k,1) = {[img_dir, '/',img_dir_items(j).name,',1']}
    		
                k=k+1
              

    	        end
    end
     

    
end
jobs{1}.spm.tools.dartel.warp.images{1,1}=c1f
jobs{1}.spm.tools.dartel.warp.images{1,2}=c2f
 
jobs{1}.spm.tools.dartel.warp.settings.template = 'Template_run1';

spm_jobman('run', jobs);

% end
