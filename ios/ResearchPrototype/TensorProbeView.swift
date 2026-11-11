import SwiftUI
import CoreML

// Engineering parity probe only. Input is a synthetic tensor, not a patient image.
struct TensorProbeView: View {
    @State private var result = "Noch nicht auf diesem Gerät geprüft."
    @State private var running = false

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("512px-Forschungsmodell").font(.title2)
            Text("Technischer Modelltest · kein klinischer Befund")
            Text(result).textSelection(.enabled)
            Button("Referenzeingabe prüfen") {
                running = true
                Task {
                    do {
                        result = try await Task.detached { try runTensorProbe() }.value
                    } catch {
                        result = "Prüfung fehlgeschlagen: \(error.localizedDescription)"
                    }
                    running = false
                }
            }.disabled(running)
        }.padding()
    }
}

private struct Fixture: Decodable {
    let expected_logit: Double
    let tolerance: Double
}

private func runTensorProbe() throws -> String {
    guard let modelURL = Bundle.main.url(forResource: "Mobile512Research", withExtension: "mlmodelc"),
          let tensorURL = Bundle.main.url(forResource: "gradient512", withExtension: "f32"),
          let fixtureURL = Bundle.main.url(forResource: "fixture", withExtension: "json") else {
        throw NSError(domain: "ResearchProbe", code: 1, userInfo: [NSLocalizedDescriptionKey: "Modell oder Referenzdateien fehlen im App-Bundle."])
    }
    let config = MLModelConfiguration()
    config.computeUnits = .cpuOnly // Establish CPU parity before other compute units.
    let model = try MLModel(contentsOf: modelURL, configuration: config)
    let bytes = try Data(contentsOf: tensorURL)
    guard bytes.count == 512 * 512 * MemoryLayout<Float>.size else {
        throw NSError(domain: "ResearchProbe", code: 2, userInfo: [NSLocalizedDescriptionKey: "Ungültige Tensorgröße."])
    }
    let pixels = try MLMultiArray(shape: [1, 1, 512, 512], dataType: .float32)
    bytes.copyBytes(to: pixels.dataPointer.assumingMemoryBound(to: UInt8.self), count: bytes.count)
    let input = try MLDictionaryFeatureProvider(dictionary: ["pixels": MLFeatureValue(multiArray: pixels)])
    let output = try model.prediction(from: input)
    guard let logits = output.featureValue(for: "logit")?.multiArrayValue else {
        throw NSError(domain: "ResearchProbe", code: 3, userInfo: [NSLocalizedDescriptionKey: "Logit-Ausgabe fehlt."])
    }
    let expected = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: fixtureURL))
    let actual = logits[0].doubleValue
    let error = abs(actual - expected.expected_logit)
    return "Logit: \(actual)\nReferenz: \(expected.expected_logit)\nAbweichung: \(error)\nTensorvergleich: \(error <= expected.tolerance ? "bestanden" : "abweichend")"
}
