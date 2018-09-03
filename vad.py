import wave
import webrtcvad
import collections
import sys

class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration
def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []
    for frame in frames:
        #sys.stdout.write('1' if vad.is_speech(frame.bytes, sample_rate) else '0')
        if not triggered:
            ring_buffer.append(frame)
            num_voiced = len([f for f in ring_buffer
                              if vad.is_speech(f.bytes, sample_rate)])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                #sys.stdout.write('+(%s)' % (ring_buffer[0].timestamp,))
                triggered = True
                voiced_frames.extend(ring_buffer)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append(frame)
            num_unvoiced = len([f for f in ring_buffer
                                if not vad.is_speech(f.bytes, sample_rate)])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
               # sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    #if triggered:
        #sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    sys.stdout.write('\n')
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])
def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def vadwav(filepath):
    vad_mode = 3
    f = wave.open(filepath, 'rb')
    num_channels, sample_width,audio, sample_rate = f.getnchannels(), f.getsampwidth(), f.readframes(f.getnframes()), f.getframerate()
    f.close()
    vad = webrtcvad.Vad(int(vad_mode))
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 30, 300, vad, frames)
    '''
    for i, segment in enumerate(segments):
        outvoice=os.path.join(outpath,"chunk-"+str(i)+'.wav')
        print outpath
        write_wave(outvoice, num_channels, sample_width, segment, sample_rate)
    '''
    total_segment = ''
    for i, segment in enumerate(segments):
        if i > 2:
            total_segment=total_segment+str(segment)

    f = wave.open(filepath+'.vad.wav', 'wb')
    f.setnchannels(num_channels)
    f.setsampwidth(sample_width)
    f.setframerate(sample_rate)
    f.writeframes(total_segment)
    f.close()

if __name__ == '__main__':
    vadwav('vad.wav')