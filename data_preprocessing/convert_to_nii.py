import os
import nibabel as nib
import pandas as pd

df=pd.read_csv('/no_backups/g009/data/oasis1_id.csv',header=None)
subjects=list(df[0])
print(f'***Subjects***={subjects}')
path='/no_backups/g009/data/OASIS/OASIS1/disc'
combined_path='/no_backups/g009/data/OASIS/OASIS1_3_combined/'
for i in range(1,13):
	dpath=path+str(i)
	paths=os.listdir(dpath)
	print(paths)
	for p in paths:
		if p in subjects:
                        basep=os.path.join(dpath,p)
                        print(f'basepath={basep}')
                        img_name=basep+'/RAW/'+p+'_mpr-1_anon.img'
                        print(f'filename={img_name}')
                        #im = nib.load(img_name)
                        dest=img_name.replace('.img', '.nii')
                        #nib.save(im,dest)
                        combined_path_dir=combined_path+p
                        #os.mkdir('mkdir '+combined_path_dir)
                        print(f"destination={os.path.join(combined_path_dir,dest.split('/')[-1])}")
                        os.system('cp '+dest+' '+os.path.join(combined_path_dir,dest.split('/')[-1]))
