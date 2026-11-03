import Foundation
import Vision

struct OCRResult: Codable {
    let path: String
    let text: String
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
            let text = (request.results ?? [])
                .compactMap { $0.topCandidates(1).first?.string }
                .joined(separator: "\n")
            results.append(OCRResult(path: path, text: text, error: nil))
        } catch {
            let detail = error as NSError
            results.append(OCRResult(path: path, text: "", error: "\(detail.domain) \(detail.code): \(detail.userInfo)"))
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
