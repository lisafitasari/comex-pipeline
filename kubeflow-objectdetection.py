#!/usr/bin/env python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import kfp
from kfp import components
from kfp import dsl
from kfp import gcp
import os

os.system('gsutil cp gs://kubeflow-pipelinev1/component/training_component_latest.yaml .')
os.system('gsutil cp gs://kubeflow-pipelinev1/component/validation_component_latest.yaml .')

kubeflow_tf_training_op = components.load_component_from_file('training_component_latest.yaml')
kubeflow_tf_validation_op = components.load_component_from_file('validation_component_latest.yaml')

@dsl.pipeline(
    name='Comex object detection pipeline',
    description=''
)
def kubeflow_training(
    data_file='gs://images_pama/txt/class_ch.txt',
    image_data_pack='gs://images_pama/center_housing.tar.gz',
    parser='simple',
    skip=False,
    num_epochs='2000',
    gcs_weight_path='gs://images_pama/model/class_ch_model_frcnn.hdf5',
    number_of_rois='32',
    network='resnet50',
    prediction_dir='gs://images_pama/config.pickle',
):
    # set the flag to use GPU trainer
    use_gpu = True

    training = kubeflow_tf_training_op(
        training_data_file=data_file,
        training_image_pack=image_data_pack,
        parser=parser,
        skip=skip,
        num_epochs=num_epochs,
        gcs_weight_path=gcs_weight_path
    ).apply(gcp.use_gcp_secret('user-gcp-sa'))

    validation = kubeflow_tf_validation_op(
        validation_data_file=data_file,
        number_of_rois=number_of_rois,
        network=network,
        prediction_dir=prediction_dir,
    ).apply(gcp.use_gcp_secret('user-gcp-sa'))


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(kubeflow_training, __file__ + '.zip')