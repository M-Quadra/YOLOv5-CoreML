//
//  CameraViewController.swift
//  YOLOv5-CoreML
//
//  Created by m_quadra on 2022/7/15.
//

import UIKit
import AVFoundation
import Vision

class CameraViewController: UIViewController {

    @IBOutlet weak var previewView: UIView!
    
    private let queue = DispatchQueue(label: UUID().uuidString)
    private let session = AVCaptureSession()
    private var previewLayer:AVCaptureVideoPreviewLayer!
    private var cameraPosition: AVCaptureDevice.Position = .back
    
    private let lock = NSLock()
    private lazy var detector = Detector()
    private var labels = [UILabel]()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        self.session.sessionPreset = .medium
        self.session.sessionPreset = .high
        self.previewLayer = AVCaptureVideoPreviewLayer(session: self.session)
        self.previewView.layer.addSublayer(self.previewLayer)
        
        self.setupSessionOutput()
        self.setupSessionInput()
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        
        self.previewLayer.frame = self.previewView.bounds
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        
        self.startSession()
    }
    
    override func viewDidDisappear(_ animated: Bool) {
        super.viewDidDisappear(animated)
        
        self.stopSession()
    }
}

// MARK: - Private
private extension CameraViewController {
    
    func setupSessionOutput() {
        self.queue.async { [weak self] in
            guard let self = self else { return }
            
            self.session.beginConfiguration()
            
            let opt = AVCaptureVideoDataOutput()
            opt.videoSettings = [
                (kCVPixelBufferPixelFormatTypeKey as String): kCVPixelFormatType_32RGBA,
            ]
            opt.setSampleBufferDelegate(self, queue: self.queue)
            
            if !self.session.canAddOutput(opt) { return }
            self.session.addOutput(opt)
            
            self.session.commitConfiguration()
        }
    }
    
    func setupSessionInput() {
        self.queue.async { [weak self] in
            guard let self = self else { return }
            guard let device = AVCaptureDevice.DiscoverySession(
                deviceTypes: [.builtInWideAngleCamera],
                mediaType: .video,
                position: self.cameraPosition
            ).devices.first else { return }
            
            self.session.beginConfiguration()
            
            for ipt in self.session.inputs {
                self.session.removeInput(ipt)
            }
            
            do {
                let ipt = try AVCaptureDeviceInput(device: device)
                if !self.session.canAddInput(ipt) { return }
                self.session.addInput(ipt)
            } catch {
                print(error)
                return
            }
            
            self.session.commitConfiguration()
        }
    }
    
    func startSession() {
        self.queue.async { [weak self] in
            guard let self = self else { return }
            self.session.startRunning()
        }
    }
    
    func stopSession() {
        self.queue.async { [weak self] in
            guard let self = self else { return }
            self.session.stopRunning()
        }
    }
    
    func refreshLabels(sampleBuffer: CMSampleBuffer) {
        if !self.lock.try() { return }
        defer { self.lock.unlock() }
        
        guard let imgBuffer = sampleBuffer.imageBuffer else { return }
        guard let detector = detector else { return }
        
        let imgSize = CGSize(
            width: CVPixelBufferGetHeight(imgBuffer),
            height: CVPixelBufferGetWidth(imgBuffer)
        )
        
        let previewSize = self.previewLayer.preferredFrameSize()
        let scale = min(
            previewSize.width / imgSize.width,
            previewSize.height / imgSize.height
        )
        let w = imgSize.width * scale
        let h = imgSize.height * scale
        let frame = CGRect(
            x: (previewSize.width - w)/2,
            y: (previewSize.height - h)/2,
            width: w, height: h
        )
        
        let orient: CGImagePropertyOrientation = self.cameraPosition == .back ? .right : .up
        var results = [Detector.Result]()
        if #available(iOS 14.0, *) {
            results = detector.inference(sampleBuffer, orientation: orient, frame: frame)
        } else {
            results = detector.inference(imgBuffer, orientation: orient, frame: frame)
        }
        
        DispatchQueue.main.async {
            while self.labels.count < results.count {
                let lab = UILabel()
                lab.alpha = 0.6
                lab.layer.borderColor = UIColor.white.cgColor
                lab.layer.borderWidth = 3
                lab.layer.cornerRadius = 6
                lab.textColor = .white
                lab.textAlignment = .center
                lab.numberOfLines = 0
                lab.adjustsFontSizeToFitWidth = true
                
                self.previewView.addSubview(lab)
                self.labels.append(lab)
            }
            while self.labels.count > results.count {
                guard let lab = self.labels.popLast() else { break }
                lab.removeFromSuperview()
            }
            
            for (i, lab) in self.labels.enumerated() {
                lab.frame = results[i].boundingBox
                lab.text = (results[i].labels.first?.identifier ?? "") + String(format: ": %.2f", results[i].confidence)
            }
        }
    }
}

// MARK: - Action
extension CameraViewController {
    
    @IBAction func switchBtnAction(_ sender: Any) {
        self.cameraPosition = self.cameraPosition == .back ? .front : .back
        self.setupSessionInput()
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBuffer Delegate
extension CameraViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        DispatchQueue.global().async {
            self.refreshLabels(sampleBuffer: sampleBuffer)
        }
    }
}
