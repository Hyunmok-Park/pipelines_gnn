import kfp.dsl as dsl
from kubernetes import client as k8s_client


def PreprocessOp():
    return dsl.ContainerOp(
        name="generate graph data",
        image="10.161.31.82:5000/phm:0.1-gnn-preprocess",
        command= [
            "sh", "run_preprocess_container.sh"
        ],
        file_outputs={
            'train_data': 'train_data.p',
            'val_data': 'val_data.p',
        }
    )

def TrainOp(train_data, val_data, vop):
    return dsl.ContainerOp(
        name="training pipeline",
        image="10.161.31.82:5000/phm:0.1-gnn-train",

        command = [
              "sh", "run_train_container.sh"
          ],
        arguments=[
            train_data, val_data
        ],
        output_artifact_paths={
          'mlpipeline-metrics': 'data/mlpipeline-metrics.json'
        },
        pvolumes={"src/data": vop},
    )

def ServeOp(trainop):
    return dsl.ContainerOp(
        name="serve pipeline",
        image="10.161.31.82:5000/phm:0.1-gnn-serve",
        command = [
            "sh", "run_serve_container.sh"
        ],
        pvolumes={"src/data": trainop.pvolume},
    )

def VolumnOp():
    return dsl.PipelineVolume(
        pvc="phm-volume"
    )

@dsl.pipeline(
    name='gnn_pipeline',
    description='Probabilistic inference with graph neural network'
)

def gnn_pipeline(
):
    print('gnn_pipeline')

    vop = VolumnOp()

    dsl.get_pipeline_conf().set_image_pull_secrets([k8s_client.V1LocalObjectReference(name='regcredidc')])

    add_p = PreprocessOp()

    train_and_eval = TrainOp(
        dsl.InputArgumentPath(add_p.outputs['train_data']),
        dsl.InputArgumentPath(add_p.outputs['val_data']),
        vop
    )

    train_and_eval.after(add_p)

    serve = ServeOp(train_and_eval)

    serve.after(train_and_eval)


if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(gnn_pipeline, __file__ + '.tar.gz')
    # compiler.Compiler().compile(gnn_pipeline, __file__ + '.yaml')

