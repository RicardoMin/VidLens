import Foundation
import Vision

struct OCRRegion: Codable {
    let text: String
    let confidence: Float
    // Vision uses a lower-left origin; all VidLens boxes use top-left x/y/w/h.
    let box: [Double]
}

struct OCRResult: Codable {
    let path: String
    let text: String
    let regions: [OCRRegion]
    let error: String?
}

var results: [OCRResult] = []
for path in CommandLine.arguments.dropFirst() {
    autoreleasepool {
        do {
            let request = VNRecognizeTextRequest()
            request.recognitionLevel = .accurate
            request.recognitionLanguages = ["zh-Hans", "en-US"]
            request.usesLanguageCorrection = false
            request.automaticallyDetectsLanguage = true
            let handler = VNImageRequestHandler(url: URL(fileURLWithPath: path))
            try handler.perform([request])
            let regions = (request.results ?? []).compactMap { observation -> OCRRegion? in
                guard let candidate = observation.topCandidates(1).first else { return nil }
                let box = observation.boundingBox
                return OCRRegion(text: candidate.string, confidence: candidate.confidence,
                    box: [Double(box.minX), Double(1 - box.maxY),
                          Double(box.width), Double(box.height)])
            }
            let text = regions.map { $0.text }.joined(separator: "\n")
            results.append(OCRResult(path: path, text: text, regions: regions, error: nil))
        } catch {
            let detail = error as NSError
            results.append(OCRResult(path: path, text: "", regions: [], error: "\(detail.domain) \(detail.code): \(detail.userInfo)"))
        }
    }
}
do {
    let encoded = try JSONEncoder().encode(results)
    FileHandle.standardOutput.write(encoded)
} catch {
    FileHandle.standardError.write(Data(error.localizedDescription.utf8))
    exit(1)
}
