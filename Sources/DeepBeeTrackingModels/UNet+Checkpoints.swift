import Checkpoints
import TensorFlow

extension Conv2D {
  func save(prefix: String, to tensors: inout [String: Tensor<Scalar>]) {
    tensors[prefix + "/filter"] = filter
    tensors[prefix + "/bias"] = bias
  }

  mutating func load(_ loader: CheckpointReader, prefix: String) {
    filter = Tensor(loader.loadTensor(named: prefix + "/filter"))
    bias = Tensor(loader.loadTensor(named: prefix + "/bias"))
  }
}

extension BatchNorm {
  func save(prefix: String, to tensors: inout [String: Tensor<Scalar>]) {
    tensors[prefix + "/offset"] = offset
    tensors[prefix + "/scale"] = scale
  }

  mutating func load(_ loader: CheckpointReader, prefix: String) {
    offset = Tensor(loader.loadTensor(named: prefix + "/offset"))
    scale = Tensor(loader.loadTensor(named: prefix + "/scale"))
  }
}

extension UNet.ConvBlock {
  func save(prefix: String, to tensors: inout [String: Tensor<Float>]) {
    conv.save(prefix: prefix + "/conv", to: &tensors)
    bn.save(prefix: prefix + "/bn", to: &tensors)
  }

  mutating func load(_ loader: CheckpointReader, prefix: String) {
    conv.load(loader, prefix: prefix + "/conv")
    bn.load(loader, prefix: prefix + "/bn")
  }
}

extension UNet {
  public var tensors: [String: Tensor<Float>] {
    var r: [String: Tensor<Float>] = [:]
    encoderConv1.save(prefix: "/encoderConv1", to: &r)
    encoderConv2.save(prefix: "/encoderConv2", to: &r)
    encoderConv3.save(prefix: "/encoderConv3", to: &r)
    encoderConv4.save(prefix: "/encoderConv4", to: &r)
    decoderConv1.save(prefix: "/decoderConv1", to: &r)
    decoderConv2.save(prefix: "/decoderConv2", to: &r)
    decoderConv3.save(prefix: "/decoderConv3", to: &r)
    decoderConv4.save(prefix: "/decoderConv4", to: &r)
    return r
  }

  public mutating func load(_ loader: CheckpointReader) {
    encoderConv1.load(loader, prefix: "/encoverConv1")
    encoderConv2.load(loader, prefix: "/encoverConv2")
    encoderConv3.load(loader, prefix: "/encoverConv3")
    encoderConv4.load(loader, prefix: "/encoverConv4")
    decoderConv1.load(loader, prefix: "/encoverConv1")
    decoderConv2.load(loader, prefix: "/encoverConv2")
    decoderConv3.load(loader, prefix: "/encoverConv3")
    decoderConv4.load(loader, prefix: "/encoverConv4")
  }
}
