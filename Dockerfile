FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.4-cuda10.1-py3

ADD . /

WORKDIR / 

RUN pip install pkg/cftime-1.4.1-cp37-cp37m-manylinux2014_x86_64.whl
RUN pip install pkg/netCDF4-1.5.6-cp37-cp37m-manylinux1_x86_64.whl

CMD ["sh", "run.sh"]