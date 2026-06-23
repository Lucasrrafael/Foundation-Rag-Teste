//
//  Item.swift
//  TesteFoundation
//
//  Created by Lucas Rafael Dias e Dias on 23/06/26.
//

import Foundation
import SwiftData

@Model
final class Item {
    var timestamp: Date
    
    init(timestamp: Date) {
        self.timestamp = timestamp
    }
}
