# default is colmap double. uses directory "colmap"

workdir=$1
datatype=$2 # blender, hypernerf, llff, phystrack
export CUDA_VISIBLE_DEVICES=0
rm -rf $workdir/sparse_
rm -rf $workdir/image_colmap
python scripts/"$datatype"2colmap.py $workdir double
rm -rf $workdir/colmap
rm -rf $workdir/colmap/sparse/0

mkdir $workdir/colmap
cp -r $workdir/image_colmap $workdir/colmap/images
cp -r $workdir/sparse_ $workdir/colmap/sparse_custom

# 각 이미지마다 feature 뽑음
colmap feature_extractor --database_path $workdir/colmap/database.db --image_path $workdir/colmap/images  --SiftExtraction.max_image_size 4096 --SiftExtraction.max_num_features 16384 --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1

# 데이터베이스에서 camera parameter gt로 교체
python database.py --database_path $workdir/colmap/database.db --txt_path $workdir/colmap/sparse_custom/cameras.txt

# 위에서 찾은 모든 image의 feature을 모아서 nC2로 매칭. 매칭이 많이 된 feature가 있고 아닌게 있는데, 각 이미지가 feature를 몇개 보는지는 전부 다름
colmap exhaustive_matcher --database_path $workdir/colmap/database.db
mkdir -p $workdir/colmap/sparse/0

# convert.py에서와 달리 camera parameter given일때는 mapper가 아닌 point_triangulator 사용
colmap point_triangulator --database_path $workdir/colmap/database.db --image_path $workdir/colmap/images --input_path $workdir/colmap/sparse_custom --output_path $workdir/colmap/sparse/0 --clear_points 1

mkdir -p $workdir/colmap/dense/0

# 카메라 파라미터를 기반으로 image undistortion. matching된 point를 많이 못보더라도 camera parameter만 있으면 undistortion 됨
colmap image_undistorter --image_path $workdir/colmap/images --input_path $workdir/colmap/sparse/0 --output_path $workdir/colmap/dense/0

# 각 이미지별로 depth map 등을 뽑아내야 하는데, neighboring image를 찾을 수 없으면 (feature를 공유하는 등) skip됨
# MODIFIED: placeholder min/max
colmap patch_match_stereo --workspace_path $workdir/colmap/dense/0 --PatchMatchStereo.depth_min 0.1 --PatchMatchStereo.depth_max 50.0

# depth 정보들을 바탕으로 최종적인 ply file을 얻어냄
colmap stereo_fusion --workspace_path $workdir/colmap/dense/0 --output_path $workdir/colmap/dense/0/fused.ply
