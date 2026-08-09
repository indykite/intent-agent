# Running gopy

https://github.com/go-python/gopy

https://pypi.org/project/PyBindGen/ - latest release 2022-03-20

```sh
go install golang.org/x/tools/cmd/goimports@latest
# go install github.com/go-python/gopy@latest
# fails with tokeninternal.go:78:9: invalid array length -delta * delta (constant -256 of type int64)
# https://github.com/go-python/gopy/issues/397
go install github.com/go-python/gopy@master
go mod init dummy.com/dum
go get github.com/go-python/gopy/_examples/hi
source .venv/bin/activate
gopy build -output=out -vm=python3 github.com/go-python/gopy/_examples/hi
python3 -c "from out import hi"
# [1]    41839 segmentation fault  python3 -c "from out import hi"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.
python3 -c "from out import hi"
# [1]    41839 segmentation fault  python3 -c "from out import hi"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:out
python3 -c "from out import hi"
# [1]    41839 segmentation fault  python3 -c "from out import hi"
cd out
make
cd ..
python3 -c "from out import hi"
# [1]    42239 segmentation fault  python3 -c "from out import hi"
```
