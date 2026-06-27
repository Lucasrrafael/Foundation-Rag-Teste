//
//  QwenGenerator.swift
//  TesteFoundation
//

import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

enum QwenGeneratorError: LocalizedError {
    case simulatorNotSupported
    case modelNotLoaded

    var errorDescription: String? {
        switch self {
        case .simulatorNotSupported:
            return "O MLX requer um iPhone físico com GPU Metal. O Simulador não é suportado."
        case .modelNotLoaded:
            return "O modelo Qwen ainda não foi carregado."
        }
    }
}

@MainActor
final class QwenGenerator {
    static let shared = QwenGenerator()

    private let modelConfiguration = ModelConfiguration(
        id: "mlx-community/Qwen2.5-3B-Instruct-4bit"
    )

    private var modelContainer: ModelContainer?
    private(set) var downloadProgress: Double = 0
    private(set) var isLoaded = false

    private init() {}

    func loadModelIfNeeded(onProgress: @escaping @Sendable (Double) -> Void = { _ in }) async throws {
        #if targetEnvironment(simulator)
        throw QwenGeneratorError.simulatorNotSupported
        #else
        guard modelContainer == nil else { return }

        Memory.cacheLimit = 20 * 1024 * 1024

        let container = try await #huggingFaceLoadModelContainer(
            configuration: modelConfiguration
        ) { progress in
            let fraction = progress.fractionCompleted
            Task { @MainActor in
                self.downloadProgress = fraction
                onProgress(fraction)
            }
        }

        modelContainer = container
        isLoaded = true
        downloadProgress = 1
        #endif
    }

    func generate(systemPrompt: String, userPrompt: String) async throws -> String {
        #if targetEnvironment(simulator)
        throw QwenGeneratorError.simulatorNotSupported
        #else
        try await loadModelIfNeeded()

        guard let modelContainer else {
            throw QwenGeneratorError.modelNotLoaded
        }

        let parameters = GenerateParameters(
            maxTokens: 512,
            temperature: 0.3
        )

        let session = ChatSession(
            modelContainer,
            instructions: systemPrompt,
            generateParameters: parameters
        )

        return try await session.respond(to: userPrompt)
        #endif
    }
}
