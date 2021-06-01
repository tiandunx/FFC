ip=127.0.0.1 # your ip address
port= # int port number
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
world_size=8
exe=main.py
queue_size= # #total_identities * ratio
net_type=r50 # if 224 x 224, use r50, for 112 x 112 input, use ir50 or mobile
log_file=${net_type}_ms_0.01id.log
saved_dir=${net_type}_ms_0.01id
if [ ! -d ${saved_dir} ]; then
  mkdir -p ${saved_dir}
fi
feat_dim=512
batch_size=64
loss_type=AM
end=$[ ${world_size} - 1]
snapshot="path to your snapshot, you may use arbitary string if you wish to train from scratch"
for local_rank in $(seq 0 ${end})
    do
    global_rank=${local_rank}
    if [ ${local_rank} -eq 0 ];then
        nohup python ${exe} ${ip} ${port} ${local_rank} ${global_rank}  ${world_size} ${saved_dir} --net_type=${net_type} --queue_size=${queue_size} --feat_dim=${feat_dim} --batch_size=${batch_size} --loss_type=${loss_type} --pretrained_model_path=${snapshot} > ${log_file} 2>&1 &
    else
        nohup python ${exe} ${ip} ${port} ${local_rank} ${global_rank} ${world_size} ${saved_dir} --net_type=${net_type} --queue_size=${queue_size} --feat_dim=${feat_dim} --batch_size=${batch_size} --loss_type=${loss_type} --pretrained_model_path=${snapshot} > /dev/null 2>&1 &
    fi
done
