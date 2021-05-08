## Replication Instructions

1. Spin up [AWS Deep Learning AMI (Ubuntu 18.04 OS) Version 43.0](https://aws.amazon.com/marketplace/pp/B07Y43P7X5). To access a gpu, use a g3.4x large infrastructure, and to replicate our performance results, use a g3.16xlarge.

2. Type the following in your terminal (assumes your working ssh key for aws is `~/.ssh/CS205-key.pm`)
`chmod 0400 ~/.ssh/CS205-key.pem` 

3. Type into terminal (Note that we use the public dns, not public ip)
`ssh -L localhost:8888:localhost:8888 -i ~/.ssh/CS205-key.pem ubuntu@<Your public instance DNS>`

4. Next steps are on the machine. Pull the git repo:
`git clone https://github.com/msbutler/cs205final`

5. To run the performance analysis code:
Run THIS

6. To play with the jupyter notebooks:
   `jupyter notebook`

   - and copy and paste the url printed to your consol into your browser. looks kinda like:
    `http://localhost:8888/?token=c65d0f53c262b4e45790a4f2e06023ded15c839ba51e7e2e`

   - navigate to the repo! use the `conda_tensorflow_p37` kernel


Instructions loosely base based on [this](https://aws.amazon.com/blogs/machine-learning/get-started-with-deep-learning-using-the-aws-deep-learning-ami/)