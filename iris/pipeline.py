import kfp
import kfp.components as comp
from kfp import dsl


def print_op(msg):
    """Print a messages"""
    return dsl.ContainerOp(
        name='Print',
        image='alpine:3.6',
        command=['echo', msg]
    )


@dsl.pipeline(
    name='iris',
    description='iris test'
)
def pipeline():
    add_p = dsl.ContainerOp(
        name="load iris data pipeline",
        image="cocoriel/iris-load:v0.1",
        arguments=[
            '--data_path', './Iris.csv'
        ],
        file_outputs={'iris': '/iris.csv'}
    )

    ml = dsl.ContainerOp(
        name="training pipeline",
        image="cocoriel/iris-training-eval:v0.1",
        arguments=[
            '--data', add_p.outputs['iris']
        ],
        file_outputs={
            'accuracy': '/accuracy.json',
            'mlpipeline-metrics': '/mlpipeline-metrics.json'
        }
    )

    ml.after(add_p)
    baseline = 0.7
    with dsl.Condition(ml.outputs['accuracy'] > baseline) as check_condition:
        print_op("accuracy는 {}로 accuracy baseline인 {}보다 큽니다!".format(ml.outputs['accuracy'], baseline))
    with dsl.Condition(ml.outputs['accuracy'] < baseline) as check_condition:
        print_op("accuracy는 {}로 accuracy baseline인 {}보다 작습니다!".format(ml.outputs['accuracy'], baseline))


if __name__ == "__main__":
    import kfp.compiler as compiler

    compiler.Compiler().compile(soojin_pipeline, __file__ + ".tar.gz")
