PR_USER=$1
PR_BRANCH=$2

git remote add $1 git@github.com:$1/onnx-web.git
git fetch $1
git push gitlab refs/remotes/$1/$2:refs/heads/$1-$2
