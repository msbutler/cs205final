## Michael Setup Instructions
(based on https://aws.amazon.com/blogs/machine-learning/get-started-with-deep-learning-using-the-aws-deep-learning-ami/)
1. spinup the ubuntu 18.04 ML ami using a g3.4x large (called  AWS Deep Learning AMI (Ubuntu 18.04))
2. type the following in your terminal
`chmod 0400 ~/.ssh/CS205-key.pem`
3. type into terminal
`ssh -L localhost:8888:localhost:8888 -i ~/.ssh/CS205-key.pem ubuntu@<Your instance DNS>`
4. type:
`jupyter notebook`
5. use the `conda_tensorflow2_latest_p37` kernel


## training pipeline
for each train zip file
 - pull a train zip file from s3, unzip it, train the model, discard the data

 ```
 aws s3 cp s3://205final/Data/Train-20210430T115344Z-001.zip Data/
 unzip Data/Train-20210430T115344Z-001.zip
 python train.py
 rmdir Train -r
 ```

 unzipped Train-001 and Train-002. But not Train-003