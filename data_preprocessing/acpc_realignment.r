#############################################
# R utility code to perform acpc realignment
#############################################

library(spm12r)
mean_img='~/.shashanks/MATLAB/spm12/canonical/avg152T1.nii'

path='/no_backups/g009/data/OASIS/OASIS1_3_combined'
folders = dir(path)

for ( i in 1:length(folders)){
img_dir_path=paste(path,'/',folders[i],sep='')
#print(img_dir_path)
files=dir(img_dir_path)
#print(files)
for(j in 1:length(files)){ 
#print(files[j])
if (startsWith(files[j],'sub-OAS3') | startsWith(files[j],'OAS1_'))
{ 
img=paste(img_dir_path,'/',files[j],sep='') 
print(img)
acpc_reorient(infiles=c(mean_img,img),modality='T1',spmdir=spm_dir())
}
}

}

