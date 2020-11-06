% Perform normalization using the dartel template followed by modulation, smoothing 

addpath('/no_backups/g009/data/OASIS/')
addpath('~/.shashanks/MATLAB/spm12')

data_dir='/no_backups/g009/data/OASIS/OASIS1_3_combined';
cd(data_dir);
dir_items = dir(data_dir);

clear matlabbatch;
for i = 1:length(dir_items)
if startsWith(dir_items(i).name,'.')
    	continue
    end
%matlabbatch{1}.spm.tools.dartel.mni_norm.template(1) = {[data_dir, '/',dir_items(i).name,'/RAW/','Template_run1_6.nii']}
matlabbatch{1}.spm.tools.dartel.mni_norm.template(1) = {[data_dir, '/',dir_items(i).name,'/','Template_run1_6.nii']}
break
end
disp(data_dir)
disp(length(dir_items))
k=1
l=1
for i = 1:length(dir_items);
    
    if startsWith(dir_items(i).name,'.')
    	
        continue
    end
    img_dir = [data_dir, '/',dir_items(i).name]
	
	img_dir_items=dir(img_dir)
	for j=1:length(img_dir_items);
		if startsWith(img_dir_items(j).name,'.')
		disp('ignore this file')
    		disp(img_dir_items(j).name)
                continue;
    	        end
        disp(img_dir_items(j).name)
    	if startsWith(img_dir_items(j).name,'u_rc1') 
    		
    		
    		disp([img_dir, '/',img_dir_items(j).name,',1'])
                f=[img_dir, '/',img_dir_items(j).name]
    		matlabbatch{1}.spm.tools.dartel.mni_norm.data.subj(l).flowfield{1} = f;  %[{[img_dir, '/',img_dir_items(j).name]}];
                l=l+1 
        end 

	if startsWith(img_dir_items(j).name,'c1') 
    		
    		
    		disp([img_dir, '/',img_dir_items(j).name,',1'])
                im = cellstr([img_dir, '/',img_dir_items(j).name])
    		matlabbatch{1}.spm.tools.dartel.mni_norm.data.subj(k).images = im;  %[img_dir, '/',img_dir_items(j).name];
                k=k+1

    	end
    end
 

end

disp(matlabbatch{1}.spm.tools.dartel.mni_norm.data.subj(1))                     
disp(matlabbatch{1}.spm.tools.dartel.mni_norm.data.subj(2))
disp(matlabbatch{1}.spm.tools.dartel.mni_norm.data.subj(3))
disp(matlabbatch{1}.spm.tools.dartel.mni_norm.data.subj(2290))

disp(matlabbatch{1}.spm.tools.dartel.mni_norm.data.subj(k-1))

matlabbatch{1}.spm.tools.dartel.mni_norm.preserve = 1; %modulation
matlabbatch{1}.spm.tools.dartel.mni_norm.fwhm = [4 4 4];%smoothing kernel size % smoothing 4mm kernel

spm_jobman('run', matlabbatch);
