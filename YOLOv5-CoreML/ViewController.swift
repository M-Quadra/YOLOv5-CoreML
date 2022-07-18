//
//  ViewController.swift
//  YOLOv5-CoreML
//
//  Created by m_quadra on 2022/7/12.
//

import UIKit

class ViewController: UIViewController {
    
    @IBOutlet weak var previewImgView: UIImageView!
    @IBOutlet weak var detectBtn: UIButton!
    
    private lazy var detector = Detector()
    private var labels = [UILabel]()
    
    override func viewDidLoad() {
        super.viewDidLoad()
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        
        self.detectBtn.sendActions(for: .touchUpInside)
    }
}

// MARK: - UIImagePickerController Delegate
extension ViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        defer { picker.dismiss(animated: true) }
        
        guard let img = (info[.editedImage] ?? info[.originalImage]) as? UIImage else { return }
        self.previewImgView.image = img
        self.detectBtn.sendActions(for: .touchUpInside)
    }
}

// MARK: - Action
private extension ViewController {
    
    @IBAction func photosBtnAction(_ sender: Any) {
        let pickerCtrl = UIImagePickerController()
        pickerCtrl.sourceType = .savedPhotosAlbum
        pickerCtrl.delegate = self
        self.present(pickerCtrl, animated: true)
    }
    
    @IBAction func detectBtnAction(_ sender: Any) {
        guard let img = self.previewImgView.image else { return }
        
        let scale = min(
            self.previewImgView.frame.width / img.size.width,
            self.previewImgView.frame.height / img.size.height
        )
        let w = img.size.width * scale
        let h = img.size.height * scale
        let frame = CGRect(
            x: (self.previewImgView.frame.width - w)/2,
            y: (self.previewImgView.frame.height - h)/2,
            width: w, height: h
        )
        
        DispatchQueue.global().async { [weak self] in
            guard let self = self else { return }
            let results = self.detector?.inference(img, frame: frame) ?? []
            
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
                    
                    self.previewImgView.addSubview(lab)
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
}
