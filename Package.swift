// swift-tools-version:5.3

import PackageDescription

let package = Package(
  name: "DeepBeeTracking",
  products: [
    .executable(name: "TrainUNet", targets: ["TrainUNet"]),
    .library(name: "DeepBeeTrackingDatasets", targets: ["DeepBeeTrackingDatasets"]),
    .library(name: "DeepBeeTrackingModels", targets: ["DeepBeeTrackingModels"]),
  ],
  dependencies: [
    .package(url: "https://github.com/tensorflow/swift-models.git", .branch("master")),
  ],
  targets: [
    .target(
      name: "DeepBeeTrackingDatasets",
      dependencies: [
        .product(name: "Datasets", package: "swift-models"),
        .product(name: "ModelSupport", package: "swift-models"),
      ]),
    .target(
      name: "DeepBeeTrackingModels",
      dependencies: [
        .product(name: "Checkpoints", package: "swift-models"),
      ]),
    //.target(
    //  name: "DeepBeeTracking",
    //  dependencies: []),
    .target(
      name: "TrainUNet",
      dependencies: [
        "DeepBeeTrackingDatasets",
        "DeepBeeTrackingModels",
        .product(name: "ModelSupport", package: "swift-models"),
      ]),
  ]
)
