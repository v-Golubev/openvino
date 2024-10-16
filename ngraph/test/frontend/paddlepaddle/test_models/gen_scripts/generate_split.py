#
# split paddle model generator
#
import numpy as np
from save_model import saveModel
import sys


def split(name : str, x, attrs : dict):
    import paddle as pdpd
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype=x.dtype)
        out = pdpd.fluid.layers.split(node_x, num_or_sections=attrs['num_or_sections'], dim=attrs['axis'])

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])
        print("outputs: ", type(outs),len(outs))
        print("out: ", type(out), len(out))

        saveModel(name, exe, feedkeys=['x'], fetchlist=out, inputs=[x], outputs=outs, target_dir=sys.argv[1])

    return outs[0]


def split_dim_tensor(name : str, x, attrs : dict, dim):
    import paddle as pdpd
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype=x.dtype)
        dim_node = pdpd.assign(dim)
        out = pdpd.fluid.layers.split(node_x, num_or_sections=attrs['num_or_sections'], dim=dim_node)

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])
        print("outputs: ", type(outs),len(outs))
        print("out: ", type(out), len(out))

        saveModel(name, exe, feedkeys=['x'], fetchlist=out, inputs=[x], outputs=outs, target_dir=sys.argv[1])

    return outs[0]


def split_test_list_tensor(name : str, x, attrs : dict):
    import paddle as pdpd
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype=x.dtype)
        section = attrs['num_or_sections']
        section[0] = pdpd.assign(np.array((section[0],)).astype('int32'))
        out = pdpd.fluid.layers.split(node_x, num_or_sections=section, dim=attrs['axis'])

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])
        print("outputs: ", type(outs),len(outs))
        print("out: ", type(out), len(out))

        saveModel(name, exe, feedkeys=['x'], fetchlist=out, inputs=[x], outputs=outs, target_dir=sys.argv[1])

    return outs[0]


def main():
    # split
    data_types = ['float32'] #TODOD: ['bool', 'float16', 'float32', 'float64', 'int32', 'int64']
    num_or_sections = [3, [2, 3, 4], [2, 3, -1]]
    axes = [1, -2]

    idx = 1
    for t in data_types:
        for s in num_or_sections:
            for i in axes:
                pdpd_attrs = {
                    'num_or_sections': s,
                    'axis': i
                }
                data_NCHW = np.random.rand(3,9,5).astype(t)
                split("split_test{}".format(idx), data_NCHW, pdpd_attrs)
                idx+=1

    split("split_test_list", data_NCHW, {
        'num_or_sections': [4, 5],
        'axis': 1})
    split_dim_tensor("split_test_dim_int32", data_NCHW, {
        'num_or_sections': 3}, np.array([1,]).astype('int32'))
    split_dim_tensor("split_test_dim_int64", data_NCHW, {
        'num_or_sections': 3}, np.array([1,]).astype('int64'))
    split_test_list_tensor("split_test_list_tensor", data_NCHW, {
        'num_or_sections': [4, 5],
        'axis': 1})


if __name__ == "__main__":
    main()
