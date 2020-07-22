import Datasets
import Foundation

/// Downloads the bee dataset (if it's not already present), and returns its URL on the local
/// system.
internal func downloadBeeDatasetIfNotPresent() -> URL {
  let downloadDir = DatasetUtilities.defaultDirectory.appendingPathComponent(
    "bees_v1", isDirectory: true)
  let framesDir = downloadDir.appendingPathComponent("frames")
  let directoryExists = FileManager.default.fileExists(atPath: downloadDir.path)
  let contentsOfDir = try? FileManager.default.contentsOfDirectory(atPath: downloadDir.path)
  let directoryEmpty = (contentsOfDir == nil) || (contentsOfDir!.isEmpty)

  guard !directoryExists || directoryEmpty else { return framesDir }

  let remoteRoot = URL(
    string: "https://storage.googleapis.com/swift-tensorflow-misc-files/beetracking")!

  let _ = DatasetUtilities.downloadResource(
    filename: "beedata", fileExtension: "tar.gz",
    remoteRoot: remoteRoot, localStorageDirectory: downloadDir
  )

  return framesDir 
}
