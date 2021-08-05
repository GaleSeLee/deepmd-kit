#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>
#include "json.hpp/json.h"
#include <iostream>
//firstnei
//map_nlist_cpu
namespace
{
    Json::Value ParseAttributes(const std::string &attributes)
    {
        // Parse Json.
        Json::CharReaderBuilder builder;
        std::string errs;
        Json::Value parsed_json;
        std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
        bool parsed =
            reader->parse(attributes.c_str(), attributes.c_str() + attributes.size(),
                          &parsed_json, &errs);
        return parsed_json;
    }
    std::vector<int> GetVectorFromJson(Json::Value &val)
    {
        std::vector<int> result;
        result.reserve(val.size());
        for (auto a : val)
        {
            result.push_back(a.asUInt64());
        }
        return result;
    }
}

void cum_sum(
    std::vector<int> &sec,
    const std::vector<int>& n_sel)
{
    sec.resize(n_sel.size() + 1);
    sec[0]=0;
    for(int ii=1; ii<sec.size();ii++)
    {
        sec[ii]=sec[ii-1]+n_sel[ii-1];
    }

}

extern "C"
{
    int32_t custom_op_api_level = 4;
}

extern "C" poplar::program::Program ProdEnvMatA(
    poplar::Graph &graph, const std::vector<poplar::Tensor> &inputs,
    std::vector<poplar::Tensor> &outputs, 
    const std::string& attributes,
    const std::string &debugPrefix)
{
    using namespace poplar;
    input_index = 0;
    const Tensor &coord_tensor = inputs[input_index++];
    const Tensor &type_tesor = inputs[input_index++];
    const Tensor &natoms_tensor = inputs[input_index++];
    //const Tensor &box_tensor = inputs[input_index++];//useless
    //const Tensor &mesh_tensor = inputs[input_index++];// delete
    const Tensor &avg_tensor = inputs[input_index++];
    const Tensor &std_tensor = inputs[input_index++];
    const Tensor &nloc_tensor = inputs[input_index++];//modify
    const Tensor &nall_tensor = inputs[input_index++];//modify
    const Tensor &ilist_tensor = inputs[input_index++];//modify
    const Tensor &numneigh_tensor = inputs[input_index++];//modify
    const Tensor &firstneigh_tensor = inputs[input_index++];//modify

    Json::Value json=ParseAttributes(attributes);
    float rcut_a = json["rcut_a"].asFloat();
    float rcut_r = json["rcut_r"].asFloat();
    float rcut_r_smth = json["rcut_r_smth"].asFloat();
    std::vector<int> = GetVectorFromJson(json["sel_a"]);
    std::vector<int> = GetVectorFromJson(json["sel_r"]);
    std::vector<int> sec_a;
    std::vector<int> sec_r;
    cum_sum(sec_a, sel_a);
    cum_sum(sec_r,sel_r);

    vector<int> a_shape({sec_a.size()});
    Tensor sec_a_tensor = addConstant(INT, ArrayRef(a_shape,int), sec_a, "sec_a");

    int nloc = nloc_tensor.shape()[0];
    int nall = nall_tensor.shape()[0];
    int ntypes = natoms_tensor.shape()[0] - 2;
    int nsamples = coord_tensor.shape()[0];
    int nnei = sec_a.back() + sec_r.back();
    int ndescrpt = sec_a.back() * 4 + sec_r.back() * 1;

    int nei_mode = 0;
    bool b_nlist_map = false;//need to check 
    int max_nbor_size = 1024;
    int max_cpy_trial = 100;
    int mem_cpy = 256;
    int max_nnei_trial = 100;
    int mem_nnei = 256;
    int sec_a_back = sec_a.back();
    
    Tensor em_descrpt_tensor = graph.addVariable(FLOAT, {nloc, ndescrpt}, "em_descrpt");
    Tensor em_descrpt_deriv_tensor = graph.addVariable(FLOAT, {nloc, ndescrpt * 3}, "em_descrpt");

    Tensor descrpt_tensor = graph.addVariable(FLOAT, {nsamples, nloc, ndescrpt}, "descrpt");
    Tensor descrpt_deriv_tensor = graph.addVariable(FLOAT, {nsamples, nloc, ndescrpt * 3}, "descrpt_deriv");
    Tensor rij_tensor = graph.addVariable(FLOAT,{nsamples, nloc, nnei*3},"rij");
    Tensor nlist_tensor = graph.addVariable(INT,{nsamples, nloc , nnei},"nlist");
    outputs.push_back(descrpt_tensor);
    outputs.push_back(descrpt_deriv_tensor);
    outputs.push_back(rij_tensor);
    outputs.push_back(nlist_tensor);

//useless
    Tensor info_type = graph.addVariable(INT, {nsamples, nloc, max_nbor_size}, "info_type");
    Tensor info_dist = graph.addVariable(FLOAT, {nsamples, nloc, max_nbor_size}, "info_dist");
    Tensor info_index = graph.addVariable(INT, {nsamples, nloc, max_nbor_size}, "info_index");

    for (int i = 0; i < nsamples; i++)
    {
        graph.setTileMapping(sec_a_tensor,0);//sec_a
        graph.setTileMapping(coord_tensor[i], i*nloc);//posi
        graph.setTileMapping(type_tesor[i],i*nloc);//type
        auto cs1 = graph.addComputeSet(debugPrefix + "/ProdFormatVertex");
        for (int j = 0; j < nloc; j++)
        {
            auto v = graph.addVertex(cs1, poputil::templateVertex("ProdFormatVertex", FLOAT));
            graph.setTileMapping(v, i * nloc + j);
            graph.setTileMapping(nlist_tensor[i][j], i * nloc + j);//fmt_nei_idx_a
            graph.setTileMapping(firstneigh_tensor[i][j], i * nloc + j);//nei_idx_a
            graph.setTileMapping(info_type[i][j], i * nloc+j);
            graph.setTileMapping(info_dist[i][j], i * nloc + j);
            graph.setTileMapping(info_index[i][j], i * nloc + j);

            graph.connect(v["posi"], coord_tensor[i]);
            graph.connect(v["type"], type_tesor[i]);
            graph.connect(v["nei_idx_a"],firstneigh_tensor[i][j]);
            graph.connect(v["sec_a"], sec_a_tensor);
            graph.connect(v["info_type"], info_type[i][j]);
            graph.connect(v["info_dist"], info_dist[i][j]);
            graph.connect(v["info_index"], info_index[i][j]);//info_type info_dist info_index is useless
            graph.connect(v["fmt_nei_idx_a"], nlist_tensor[i][j]);
            graph.setInitialValue(v["i_idx"], j);
            graph.setInitialValue(v["rcut"], rcut_r);
        }

        auto cs2 = graph.addComputeSet(debugPrefix + "/EnvMatVertex");
        graph.setTileMapping(avg,0);//avg
        graph.setTileMapping(std,0);//std
        for (int j = 0; j < nloc; j++)
        {
            graph.setTileMapping(v, i*nloc+j);

            graph.setTileMapping(em_descrpt_tensor[j], i*nloc+j);
            graph.setTileMapping(em_descrpt_deriv_tensor[j], i*nloc+j);
            graph.setTileMapping(descrpt_tensor[i][j], i*nloc+j);
            graph.setTileMapping(descrpt_deriv_tensor[i][j], i*nloc+j);
            graph.setTileMapping(rij_tensor[i][j], i*nloc+j);

            graph.connect(v["descrpt_a"], descrpt_tensor[i][j]);
            graph.connect(v["descrpt_a_deriv"], descrpt_deriv_tensor[i][j]);
            graph.connect(v["em_descrpt_a"],em_descrpt_tensor[j]);
            graph.connect(v["em_descrpt_a_deriv"], em_descrpt_deriv_tensor[j]);
            graph.connect(v["rij_a"], rij_tensor[i]);

            graph.connect(v["std"],std);
            graph.connect(v["avg"],avg);
            graph.connect(v["posi"],coord_tensor[i]);
            graph.connect(v["type"],type_tensor[i]);
            graph.connect(v["sec_a"],sec_a_tensor);
            graph.connect(v["fmt_nlist_a"],nlist_tensor[i][j]);
            
            graph.setInitialValue(v["nem"],nnei*3);
            graph.setInitialValue(v["i_idx"], j);
            graph.setInitialValue(v["rmin"], rcut_r);
            graph.setInitialValue(v["rmax"], rcut_r_smth);

        }
    }
    return program::Sequence(
        program::Execute(cs1),
        program::Execute(cs2)
    )
}