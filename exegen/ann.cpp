// Setup TemplNet
constexpr auto dconf = DerivConfig::D12_VD1; // configure necessary derivatives
using RealT = double;

const int HIDDENLAYERSIZE = 12; // excluding (!) offset "unit"
using L1Type = LayerConfig<HIDDENLAYERSIZE, actf::TanSig>;
using L2Type = LayerConfig<1, actf::Exp>;
using NetType = TemplNet<RealT, dconf, 1, 1, L1Type, L2Type>;
QTemplWrapper<NetType> ann;
