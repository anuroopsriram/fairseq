#!/bin/zsh

job_dir=$1
split=$2
num_shards=$3
checkpoint=$4

num_shards=$(expr ${num_shards} - 1)

cp=${checkpoint##*/}

cat $job_dir/{0..$num_shards}_ref.word-${cp}-$split.txt >! $job_dir/ref.word-${cp}-$split.txt
cat $job_dir/{0..$num_shards}_hypo.word-${cp}-$split.txt >! $job_dir/hypo.word-${cp}-$split.txt

/private/home/abaevski/sctk-2.4.10/bin/sclite -r $job_dir/ref.word-${cp}-$split.txt \
-h $job_dir/hypo.word-${cp}-$split.txt  -i rm -o all stdout >! $job_dir/report_$split