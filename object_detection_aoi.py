import os
from os.path import join

from rastervision.core.rv_pipeline import *
from rastervision.core.backend import *
from rastervision.core.data import *
from rastervision.core.analyzer import *
from rastervision.pytorch_backend import *
from rastervision.pytorch_learner import *
from rastervision.pytorch_backend.examples.utils import get_scene_info, save_image_crop


raw_uri = './code'
processed_uri = './code'


def get_config(runner, raw_uri, processed_uri, root_uri, nochip = True, test = False):
    train_ids = ['0']
    val_ids = ['0']

    def make_scene(id):
        raster_uri = join(raw_uri,'raster.xml')
        label_uri = join(processed_uri, 'labels_{}.geojson'.format(id))
        aoi_uri = join(processed_uri, 'aoi.geojson')

        raster_source = RasterioSourceConfig(
            uris=[raster_uri], channel_order=[0, 1, 2])

        vector_source = GeoJSONVectorSourceConfig(
            uri=label_uri, default_class_id=0, ignore_crs_field=True)
        label_source = ObjectDetectionLabelSourceConfig(
            vector_source=vector_source)

        return SceneConfig(
            id=id, raster_source=raster_source, label_source=label_source, aoi_uris=[aoi_uri])

    class_config = ClassConfig(names=['pool'])
  
    scene_dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=[make_scene(id) for id in train_ids],
        validation_scenes=[make_scene(id) for id in val_ids])

    chip_sz = 600
    img_sz = chip_sz

    chip_options = ObjectDetectionChipOptions(neg_ratio=1.0, ioa_thresh=0.9)

    if nochip:
        window_opts = ObjectDetectionGeoDataWindowConfig(
            method=GeoDataWindowMethod.random,
            size=chip_sz,
            size_lims=(chip_sz, chip_sz + 1),
            max_windows=10,
            clip=True,
            neg_ratio=chip_options.neg_ratio,
            ioa_thresh=chip_options.ioa_thresh,
            neg_ioa_thresh=0.2)

        data = ObjectDetectionGeoDataConfig(
            scene_dataset=scene_dataset,
            window_opts=window_opts,
            img_sz=img_sz,
            augmentors=[],
            num_workers=1)
    else:
        data = ObjectDetectionImageDataConfig(img_sz=img_sz, num_workers=1)


    backend = PyTorchObjectDetectionConfig(
        data = data,
        model=ObjectDetectionModelConfig(backbone=Backbone.resnet50),
        solver=SolverConfig(
            lr=1e-4,
            num_epochs=1,
            test_num_epochs=1,
            batch_sz=8,
            one_cycle=True),
        log_tensorboard=False,
        run_tensorboard=False,
        test_mode=test)

    predict_options = ObjectDetectionPredictOptions(
        merge_thresh=0.5, score_thresh=0.9)



    return ObjectDetectionConfig(
        root_uri=root_uri,
        dataset=scene_dataset,
        backend=backend,
        train_chip_sz=chip_sz,
        predict_chip_sz=chip_sz,
        chip_options=chip_options,
        predict_options=predict_options)
