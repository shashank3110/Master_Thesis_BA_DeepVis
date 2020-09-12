import os

path='/no_backups/g009/data/OASIS/OASIS3/'
flist = os.listdir(path)

for f in flist:
        run_path=path+f
        run=sorted(os.listdir(run_path))[-1]
        fpath= os.path.join(path,os.path.join(run_path,run))
        print(f'fpath={fpath}')
        fname=os.listdir(fpath)
        fname=[f for f in os.listdir(fpath) if '.nii' in f][0]
        print(f'filename={fname}')
        # uncomment below block to enable unzip of .gz files
        #if '.gz' in fname:
        #       os.system('gzip -d '+os.path.join(fpath,fname))
        source=os.path.join(fpath,fname)#.split('.gz')[0])
        destination=os.path.join('/no_backups/g009/data/OASIS/OASIS1_3_combined/'+f,fname)#.split('.gz')[0])
        print(f'source={source},destination={destination}')
        # print(f'source={source},'destination={destination}')
        os.system('cp '+source +' '+ destination)
