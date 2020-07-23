import Checkpoints
import DeepBeeTrackingDatasets
import DeepBeeTrackingModels
import Foundation
import ModelSupport
import TensorFlow

/// Given the segmentation of a single frame, return a batch of segmentations of tiles.
func tiles(_ frame: BeeFrameSegmentations.Segmentation) -> BeeFrameSegmentations.Segmentation {
  precondition(frame.data.shape.count == 3, "Frame should be a single multichannel image")
  let w = frame.data.shape[0]
  let h = frame.data.shape[1]
  let c = frame.data.shape[2]
  let tileSize = 120
  let xTileCount = w / tileSize
  let yTileCount = h / tileSize
  return LabeledData(collating: (0..<xTileCount).lazy.flatMap { i in
    (0..<yTileCount).lazy.map { j in
      let x = i * tileSize
      let y = j * tileSize
      return LabeledData(
        data: frame.data.slice(lowerBounds: [x, y, 0], sizes: [tileSize, tileSize, c]),
        label: frame.label.slice(lowerBounds: [x, y, 0], sizes: [tileSize, tileSize, 1]))
    }
  })
}

let segmentations = BeeFrameSegmentations()
var model = UNet()
var opt = Adam(for: model, learningRate: 1e-4)

for epoch in 0..<30 {
  print("Running epoch \(epoch)")
  for segmentation in segmentations.train {
    Context.local.learningPhase = .training
    let batch = tiles(segmentation)
    let (loss, grad) = valueWithGradient(at: model) { model -> Tensor<Float> in
      return sigmoidCrossEntropy(logits: model(batch.data), labels: Tensor<Float>(batch.label))
    }
    print("Loss: \(loss)")
    opt.update(&model, along: grad)

    let writer = CheckpointWriter(tensors: model.tensors)
    try! writer.write(to: URL(string: "checkpoints")!, name: "bee-unet")
  }
}
