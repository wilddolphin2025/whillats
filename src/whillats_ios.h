/*
 *  (c) 2025, wilddolphin2025 
 *  For WebRTCsays.ai project
 *  https://github.com/wilddolphin2025
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef WHILLATS_IOS_H_
#define WHILLATS_IOS_H_

#include "whillats.h"

#if TARGET_OS_IOS
    #import <Foundation/Foundation.h>
    #import <AVFoundation/AVFoundation.h>
#else
    #import <AppKit/AppKit.h>
    #import <AudioToolbox/AudioToolbox.h>
#endif

NS_ASSUME_NONNULL_BEGIN

typedef void (*AudioCallback)(bool success, const uint16_t *_Nullable buffer, size_t size, void *_Nullable user_data);
typedef void (*CompletionCallback)(void *_Nullable user_data);

#if TARGET_OS_IOS
@interface WhillatsSpeechSynthesizerProcessor : NSObject <AVSpeechSynthesizerDelegate>
#else
@interface WhillatsSpeechSynthesizerProcessor : NSObject
#endif

- (instancetype)initWithAudioCallback:(AudioCallback)audioCallback
                             userData:(void *)userData
                    completionCallback:(CompletionCallback)completionCallback;
- (void)synthesizeText:(NSString *)text language:(NSString *)language;
- (void)stop;

#if TARGET_OS_IOS
// Speakerphone control
- (BOOL)enableSpeakerphone;
- (BOOL)disableSpeakerphone;
#endif

// Expose the C callback context pointer
@property (nonatomic, assign, readonly) void *userData;

@end

NS_ASSUME_NONNULL_END

#endif // WHILLATS_IOS_H_ 