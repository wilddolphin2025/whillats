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

#include "whillats.h"
#import "whillats_ios.h"

#if TARGET_OS_IOS

@interface WhillatsSpeechSynthesizerProcessor ()
@property (nonatomic, strong) AVSpeechSynthesizer *synthesizer;
@property (nonatomic, strong) AVAudioEngine *audioEngine;
@property (nonatomic, strong) AVAudioConverter *converter;
@property (nonatomic, strong) AVAudioFormat *outputFormat;
@property (nonatomic, strong) AVAudioFormat *engineFormat; 
@property (nonatomic, assign) AudioCallback audioCallback;
@property (nonatomic, assign) void *userData;
@property (nonatomic, assign) CompletionCallback completionCallback;
@property (nonatomic, strong) AVAudioPlayerNode *playerNode;
@end

@implementation WhillatsSpeechSynthesizerProcessor

- (instancetype)initWithAudioCallback:(AudioCallback)audioCallback
                             userData:(void *)userData
                    completionCallback:(CompletionCallback)completionCallback {
    self = [super init];
    if (self) {
        _audioCallback = audioCallback;
        _userData = userData;
        _completionCallback = completionCallback;
        [self setupSynthesizer];
        [self setupAudioEngine];
    }
    return self;
}

- (void)setupSynthesizer {
    _synthesizer = [[AVSpeechSynthesizer alloc] init];
    _synthesizer.delegate = self;
}

- (void)setupAudioEngine {
    // Configure AVAudioSession
    AVAudioSession *session = [AVAudioSession sharedInstance];
    NSError *error;
    // Modified: Use PlayAndRecord category to support speakerphone routing
    [session setCategory:AVAudioSessionCategoryPlayAndRecord
                    mode:AVAudioSessionModeDefault
                 options:AVAudioSessionCategoryOptionDefaultToSpeaker // Prefer speakerphone
                   error:&error];
    if (error) {
        NSLog(@"Failed to set AVAudioSession category: %@", error);
        if (_audioCallback) {
            _audioCallback(false, NULL, 0, _userData);
        }
        return;
    }
    
    // Set preferred sample rate to 48 kHz
    [session setPreferredSampleRate:48000 error:&error];
    if (error) {
        NSLog(@"Failed to set preferred sample rate: %@", error);
    }
    
    // Modified: Explicitly set speakerphone as the output
    [session overrideOutputAudioPort:AVAudioSessionPortOverrideSpeaker error:&error];
    if (error) {
        NSLog(@"Failed to set speakerphone: %@", error);
    }
    
    [session setActive:YES error:&error];
    if (error) {
        NSLog(@"Failed to activate AVAudioSession: %@", error);
        if (_audioCallback) {
            _audioCallback(false, NULL, 0, _userData);
        }
        return;
    }
    
    NSLog(@"Actual sample rate: %f", [session sampleRate]);
    
    _audioEngine = [[AVAudioEngine alloc] init];
    _outputFormat = [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatInt16
                                                     sampleRate:16000
                                                       channels:1
                                                    interleaved:YES];
    
    _playerNode = [[AVAudioPlayerNode alloc] init];
    [_audioEngine attachNode:_playerNode];
    
    _engineFormat = [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatFloat32
                                                    sampleRate:48000
                                                      channels:1
                                                   interleaved:YES];
    [_audioEngine connect:_playerNode to:_audioEngine.mainMixerNode format:_engineFormat];
    
    __weak WhillatsSpeechSynthesizerProcessor *weakSelf = self;
    [_playerNode installTapOnBus:0
                      bufferSize:1024
                          format:_engineFormat
                           block:^(AVAudioPCMBuffer * _Nonnull buffer, AVAudioTime * _Nonnull when) {
        [weakSelf processBuffer:buffer];
    }];
    
    [_audioEngine prepare];
    
    [_audioEngine startAndReturnError:&error];
    if (error) {
        NSLog(@"Audio engine start failed: %@", error);
        if (_audioCallback) {
            _audioCallback(false, NULL, 0, _userData);
        }
    }
}

- (void)synthesizeText:(NSString *)text language:(NSString *)language {
    dispatch_queue_t synthesisQueue = dispatch_queue_create("com.speech.synthesis", DISPATCH_QUEUE_SERIAL);
    dispatch_async(synthesisQueue, ^{
        AVSpeechUtterance *utterance = [[AVSpeechUtterance alloc] initWithString:text];
        utterance.voice = [AVSpeechSynthesisVoice voiceWithLanguage:language];
        utterance.rate = 0.5;
        
        AVAudioPCMBuffer *silenceBuffer = [[AVAudioPCMBuffer alloc] initWithPCMFormat:self->_engineFormat
                                                                       frameCapacity:48000];
        silenceBuffer.frameLength = 48000;
        [self->_playerNode scheduleBuffer:silenceBuffer completionHandler:nil];
        [self->_playerNode play];
        
        [self.synthesizer speakUtterance:utterance];
    });
}

- (void)stop {
    [_synthesizer stopSpeakingAtBoundary:AVSpeechBoundaryImmediate];
    [_playerNode stop];
    [_audioEngine stop];
}

// New: Enable speakerphone
- (BOOL)enableSpeakerphone {
    AVAudioSession *session = [AVAudioSession sharedInstance];
    NSError *error;
    // Set the audio session category to support speakerphone
    BOOL categorySet = [session setCategory:AVAudioSessionCategoryPlayAndRecord
                                withOptions:AVAudioSessionCategoryOptionDefaultToSpeaker
                                      error:&error];
    if (!categorySet) {
        NSLog(@"Failed to set audio session category: %@", error);
        return NO;
    }
    // Activate the audio session
    BOOL activated = [session setActive:YES error:&error];
    if (!activated) {
        NSLog(@"Failed to activate audio session: %@", error);
        return NO;
    }
    // Override output to speaker
    BOOL success = [session overrideOutputAudioPort:AVAudioSessionPortOverrideSpeaker error:&error];
    if (!success) {
        NSLog(@"Failed to enable speakerphone: %@", error);
    } else {
        NSLog(@"Speakerphone enabled");
    }
    return success;
}

// New: Disable speakerphone (revert to default, e.g., earpiece)
- (BOOL)disableSpeakerphone {
    AVAudioSession *session = [AVAudioSession sharedInstance];
    NSError *error;
    // Set the audio session category to support default output
    BOOL categorySet = [session setCategory:AVAudioSessionCategoryPlayAndRecord
                                withOptions:0 // No specific options to default to earpiece
                                      error:&error];
    if (!categorySet) {
        NSLog(@"Failed to set audio session category for disable: %@", error);
        return NO;
    }
    // Activate the audio session
    BOOL activated = [session setActive:YES error:&error];
    if (!activated) {
        NSLog(@"Failed to activate audio session for disable: %@", error);
        return NO;
    }
    // Revert output to default (e.g., earpiece)
    BOOL success = [session overrideOutputAudioPort:AVAudioSessionPortOverrideNone error:&error];
    if (!success) {
        NSLog(@"Failed to disable speakerphone: %@", error);
    } else {
        NSLog(@"Speakerphone disabled");
    }
    return success;
}

#pragma mark - AVSpeechSynthesizerDelegate

- (void)speechSynthesizer:(AVSpeechSynthesizer *)synthesizer
      willSpeakRangeOfSpeechString:(NSRange)characterRange
                         utterance:(AVSpeechUtterance *)utterance {
    // Optional: Track progress
}

- (void)speechSynthesizer:(AVSpeechSynthesizer *)synthesizer
         didFinishSpeechUtterance:(AVSpeechUtterance *)utterance {
    [_playerNode stop];
    if (_completionCallback) {
        _completionCallback(_userData);
    }
}

- (void)processBuffer:(AVAudioPCMBuffer *)buffer {
    if (!buffer) {
        if (_audioCallback) {
            _audioCallback(false, NULL, 0, _userData);
        }
        return;
    }
    
    AVAudioPCMBuffer *convertedBuffer = [[AVAudioPCMBuffer alloc] initWithPCMFormat:_outputFormat
                                                                     frameCapacity:(uint32_t)(buffer.frameLength * 16000.0 / 48000.0)];
    
    if (!_converter) {
        _converter = [[AVAudioConverter alloc] initFromFormat:buffer.format toFormat:_outputFormat];
    }
    
    NSError *error;
    AVAudioConverterOutputStatus outputStatus = [_converter convertToBuffer:convertedBuffer
                                                                     error:&error
                                                        withInputFromBlock:^AVAudioBuffer * _Nullable(AVAudioPacketCount inNumberOfPackets, AVAudioConverterInputStatus * _Nonnull outStatus) {
        *outStatus = AVAudioConverterInputStatus_HaveData;
        return buffer;
    }];
    
    if (outputStatus != AVAudioConverterOutputStatus_HaveData || error) {
        NSLog(@"Buffer conversion failed: %@", error);
        if (_audioCallback) {
            _audioCallback(false, NULL, 0, _userData);
        }
        return;
    }
    
    uint32_t samplesPerFrame = 160;
    int16_t *pcmData = convertedBuffer.int16ChannelData[0];
    uint32_t totalSamples = convertedBuffer.frameLength;
    
    NSMutableArray *uint16Buffers = [NSMutableArray array];
    for (uint32_t i = 0; i < totalSamples; i += samplesPerFrame) {
        uint32_t samplesToCopy = MIN(samplesPerFrame, totalSamples - i);
        uint16_t *uint16Buffer = (uint16_t *)malloc(samplesToCopy * sizeof(uint16_t));
        for (uint32_t j = 0; j < samplesToCopy; j++) {
            uint16Buffer[j] = (uint16_t)(pcmData[i + j] + 32768);
        }
        [uint16Buffers addObject:[NSData dataWithBytesNoCopy:uint16Buffer length:samplesToCopy * sizeof(uint16_t) freeWhenDone:YES]];
    }
    
    for (NSData *bufferData in uint16Buffers) {
        if (_audioCallback) {
            _audioCallback(true, (const uint16_t *)bufferData.bytes, bufferData.length / sizeof(uint16_t), _userData);
        }
    }
}

@synthesize userData = _userData;

@end

void whillats_post_transcript(const char* text, bool is_final, const char* language) {
    NSString* t = [NSString stringWithUTF8String:text ? text : ""];
    NSString* l = [NSString stringWithUTF8String:language ? language : ""];
    if (!t) return;
    NSDictionary* info = @{@"text": t, @"final": @(is_final), @"language": l ?: @""};
    dispatch_async(dispatch_get_main_queue(), ^{
        [[NSNotificationCenter defaultCenter] postNotificationName:@"WhillatsTranscriptNotification"
                                                            object:nil
                                                          userInfo:info];
    });
}

#endif // TARGET_OS_IOS