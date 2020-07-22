import Foundation
import ModelSupport
import TensorFlow

/// A dataset of frames, with segmentation labels, from a video of bees.
public struct BeeFrameSegmentations {
  public typealias Segmentation = LabeledData<Tensor<Float>, Tensor<Bool>>

  /// The training data.
  public let train: LazyMapCollection<[(String, String)], Segmentation>

  // We could add testing/validation data here too.

  /// Creates a `BeeFrameSegmentations` using the default data, automatically downloading the
  /// data from the internet if it is not already present on the local system.
  public init() {
    let dir = downloadBeeDatasetIfNotPresent()
    self.init(directory: dir)!
  }

  /// Creates a `BeeFrameSegmentations` from the data in the given `directory`.
  ///
  /// The directory must contain a file named "train.txt" whose lines are:
  ///     RELATIVE_PATH_TO_FRAME_PNG RELATIVE_PATH_TO_LABEL_PNG
  /// where the "frame png" and "label png" are images with the same width and height and where
  /// a value of 255 in the first channel of the "label png" denotes that the corresponding pixel
  /// in the "frame png" is in the class.
  public init?(directory: URL) {
    let trainFile = directory.appendingPathComponent("train.txt")
    guard let trainContents = try? String(contentsOf: trainFile) else { return nil }
    let trainLines = trainContents.split(separator: "\n").compactMap { (line) -> (String, String)? in
      let cols = line.split(separator: " ")
      guard cols.count == 2 else { return nil }
      return (String(cols[0]), String(cols[1]))
    }
    self.train = trainLines.lazy.map { line in
      let data = Image(jpeg: directory.appendingPathComponent(line.0)).tensor
      let label = Image(jpeg: directory.appendingPathComponent(line.1)).tensor
      assert(data.rank == 3)
      assert(label.rank == 3)
      assert(data.shape[0] == label.shape[0])
      assert(data.shape[1] == label.shape[1])
      let convertedLabel = label.slice(
        lowerBounds: [0, 0, 0], upperBounds: [label.shape[0], label.shape[1], 1]) .== Tensor(255)
      return LabeledData(data: data, label: convertedLabel)
    }
  }
}
