//
//  Detector.swift
//  YOLOv5-CoreML
//
//  Created by m_quadra on 2022/7/13.
//

import Vision
import UIKit

class Detector {
    
    private let request: VNCoreMLRequest
    
    var labelConfidenceThreshold: Float = 0.2
    var resultConfidenceThreshold: Float = 0.4
    
    init?() {
        let cls = YOLOv5s.self
        guard let url = Bundle(for: cls).url(forResource: String(describing: cls), withExtension: "mlmodelc") else { return nil }
        
        let cfg = MLModelConfiguration()
        cfg.allowLowPrecisionAccumulationOnGPU = true
        guard let yolo = try? MLModel(contentsOf: url, configuration: cfg) else { return nil }
        guard let model = try? VNCoreMLModel(for: yolo) else { return nil }
        
        self.request = VNCoreMLRequest(model: model)
        self.request.imageCropAndScaleOption = .scaleFit
    }
    
    struct Result {
        var boundingBox: CGRect
        var confidence: Float
        var labels: [VNClassificationObservation]
    }
    
    func inference(_ image: UIImage, frame: CGRect) -> [Result] {
        guard let cgImg = { () -> CGImage? in
            guard let cgImg = image.cgImage else { return nil }
            if cgImg.colorSpace == CGColorSpace(name: CGColorSpace.sRGB) {
                return cgImg
            }
            
            UIGraphicsBeginImageContextWithOptions(image.size, false, image.scale)
            defer { UIGraphicsEndImageContext() }
            
            image.draw(at: .zero)
            return UIGraphicsGetImageFromCurrentImageContext()?.cgImage
        }() else { return [] }
        
        let reqHandler = VNImageRequestHandler(cgImage: cgImg)
        return self.inference(reqHandler, frame: frame)
    }
    
    @available(iOS 14.0, *)
    func inference(_ buffer: CMSampleBuffer, orientation: CGImagePropertyOrientation = .up, frame: CGRect) -> [Result] {
        let reqHandler = VNImageRequestHandler(cmSampleBuffer: buffer, orientation: orientation)
        return self.inference(reqHandler, frame: frame)
    }
    
    func inference(_ buffer: CVImageBuffer, orientation: CGImagePropertyOrientation = .up, frame: CGRect) -> [Result] {
        let reqHandler = VNImageRequestHandler(cvPixelBuffer: buffer, orientation: orientation)
        return self.inference(reqHandler, frame: frame)
    }
}

// MARK: - Private
private extension Detector {
    
    func inference(_ requestHandler: VNImageRequestHandler, frame: CGRect) -> [Result] {
        do {
            try requestHandler.perform([self.request])
        } catch {
            print(error)
            return []
        }
        
        return self.request.results?.compactMap({ it -> Result? in
            guard let result = it as? VNRecognizedObjectObservation else { return nil }
            if result.confidence < self.resultConfidenceThreshold { return nil }
            if (result.labels.first?.confidence ?? -1) < self.labelConfidenceThreshold { return nil }
            
            var bbox = result.boundingBox
            bbox.origin.y = 1 - bbox.origin.y - bbox.size.height
            bbox = VNImageRectForNormalizedRect(
                bbox,
                Int(frame.size.width),
                Int(frame.self.height)
            ).standardized
            bbox.origin.x += frame.origin.x
            bbox.origin.y += frame.origin.y
            return Result(
                boundingBox: bbox,
                confidence: result.confidence,
                labels: result.labels
            )
        }) ?? []
    }
}
