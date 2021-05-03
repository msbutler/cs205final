## Michael Setup Instructions
(based on https://aws.amazon.com/blogs/machine-learning/get-started-with-deep-learning-using-the-aws-deep-learning-ami/)
1. spinup the ubuntu 18.04 ML ami using a g3.4x large (called  AWS Deep Learning AMI (Ubuntu 18.04))
2. type the following in your terminal
`chmod 0400 ~/.ssh/CS205-key.pem`

3. type into terminal (note that we use the public dns, not public ip)
`ssh -L localhost:8888:localhost:8888 -i ~/.ssh/CS205-key.pem ubuntu@<Your public instance DNS>`

4. Next steps are on the machine. Pull the git repo:
`git clone https://github.com/msbutler/cs205final`


5. Load the training data (zipfolder) onto the machine (you might need to use public ip, instead of dns), onto the repo:
`scp -i ./CS205-key.pem ./path/to/trainingzipfile <username>@<public-dns>:/`

6. Unzip the training data, and ensure the root folder is named `Train`:

`unzip {training zip file}`

7. verify that the root of the vm contains the `Train` and `cs205final` directories.

8. launch jupyter:
`jupyter notebook`

9. copy and paste the url printed to your consol into your browser. looks kinda like:
`http://localhost:8888/?token=c65d0f53c262b4e45790a4f2e06023ded15c839ba51e7e2e`

10. navigate to the repo! use the `conda_tensorflow_p37` kernel

11. verify that you're connected to the gpu. In a jupyter cell
```
with tf.Session() as sess:
    devices = sess.list_devices()
    for d in devices:
        print(d)
```

12. To push updates to the repo from the vm, remember to save your work in the jupyter ide. then ctrl+c from the terminal running jupyter. then follow the standard git add,commit,push commands.
