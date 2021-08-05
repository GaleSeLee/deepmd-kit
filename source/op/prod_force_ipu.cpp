#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

#include <iostream>

extern "C"
{
    int32_t custom_op_api_level = 4;
}

extern "C" poplar::program::Program ProdForceSeA(
    poplar::Graph &graph, const std::vector<poplar::Tensor> &inputs,
    std::vector<poplar::Tensor> &outputs, const std::string &debugPrefix)
{
    using namespace poplar;
    int input_index=0;
    const Tensor& net_deriv_tensor  = inputs[input_index++];
    const Tensor& in_deriv_tensor   = inputs[input_index++];
    const Tensor& nlist_tensor      = inputs[input_index++];
    const Tensor& nloc_tensor       = inputs[input_index++];
    const Tensor& nall_tensor       = inputs[input_index++];

    int nall = nall_tensor.shape()[0];
    int nloc = nloc_tensor.shape()[0];
    int ndescrpt = net_deriv_tensor.shape()[1] / nloc;
    int nframes = net_deriv_tensor.shape()[0];
    int nnei = nlist_tensor.shape()[1] / nloc;


    Tensor force_tensor = graph.addVariable(FLOAT, {nframes, 3*nall}, "force");
    outputs.push_back(force_tensor);

    auto cs1 = graph.addComputeSet(debugPrefix + "/ProdForceVertex_1");
    for(int i=0;i<nframes;i++)
    {
        for(int j=0;j<nloc;j++)
        {
            auto v = graph.addVertex(cs1, poputil::templateVertex("ProdForceVertex_1", FLOAT));
            graph.setTileMapping(v, i*nloc+j);
            graph.setTileMapping(net_deriv_tensor[i].reshape({nloc, ndescrpt})[j] , i*nloc+j  );
            graph.setTileMapping(in_deriv_tensor[i].reshape({nloc, ndescrpt*3})[j], i*nloc+j  );
            graph.setTileMapping(force_tensor[i].reshape({nall, 3})[j], i*nloc+j);
            
            graph.connect(v["net_deriv"], net_deriv_tensor[i].reshape({nloc, ndescrpt})[j]);
            graph.connect(v["env_deriv"], in_deriv_tensor[i].reshape({nloc, ndescrpt*3})[j]);
            graph.connect(v["force"], force_tensor[i].reshape({nall, 3})[j]);

            graph.setInitialValue(v["nall"], nall);
            graph.setInitialValue(v["nnei"], nnei);
            graph.setInitialValue(v["nloc"], nloc);
            graph.setInitialValue(v["ndescrpt"], ndescrpt);

        }

    }


    auto cs2 = graph.addComputeSet(debugPrefix + "/ProdForceVertex_2");
    for(int i=0;i<nframes;i++)
    {
        auto v = graph.addVertex(cs2, poputil::templateVertex("ProdForceVertex_2", FLOAT));
        graph.setTileMapping(v,i);
        graph.setTileMapping(nlist_tensor[i], i);
        graph.setTileMapping(force_tensor[i],i);

        graph.connect(v["net_deriv"], net_deriv_tensor[i]);
        graph.connect(v["env_deriv"], in_deriv_tensor[i]);
        graph.connect(v["force"], force_tensor[i]);
        graph.connect(v["nlist"],nlist_tensor[i]);


        graph.setInitialValue(v["nall"], nall);
        graph.setInitialValue(v["nnei"], nnei);
        graph.setInitialValue(v["nloc"], nloc);
        graph.setInitialValue(v["ndescrpt"], ndescrpt);
    }

    return program::Sequence(
        program::Execute(cs1),
        program::Execute(cs2)
    )   ;

}
