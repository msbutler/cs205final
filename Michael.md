## Michael Setup Instructions
(based on https://aws.amazon.com/blogs/machine-learning/get-started-with-deep-learning-using-the-aws-deep-learning-ami/)
1. spinup the ubuntu 18.04 ML ami using a g3.4x large (called  AWS Deep Learning AMI (Ubuntu 18.04))
2. type the following in your terminal
`chmod 0400 ~/.ssh/CS205-key.pem`

3. type into terminal (note that we use the public dns, not public ip)
`ssh -L localhost:8888:localhost:8888 -i ~/.ssh/CS205-key.pem ubuntu@<Your public instance DNS>`

4. Next steps are on the machine. Load the training data (zipfolder) onto the machine (you might need to use public ip, instead of dns)
`scp -i ./CS205-key.pem ./path/to/trainingzipfile <username>@<public-dns>:/`

5. Unzip the training data, and ensure the root folder is named `Train`:

`unzip {training zip file}`

5. type:
`jupyter notebook`

6. use the `conda_tensorflow_p37` kernel

7. verify that you're connected to the gpu. In a jupyter cell
```
with tf.Session() as sess:
    devices = sess.list_devices()
    for d in devices:
        print(d)
```


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