# NEON AI (TM) SOFTWARE, Software Development Kit & Application Development System
# All trademark and other rights reserved by their respective owners
# Copyright 2008-2021 Neongecko.com Inc.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions
#    and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
#    and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from jarbas_stt_plugin_vosk import VoskKaldiSTT
from ovos_utils.log import LOG
import unittest
import os
from jiwer import wer
from timeit import default_timer as timer
import re

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_PATH_EN = os.path.join(ROOT_DIR, "test_audio/en")
TEST_PATH_FR = os.path.join(ROOT_DIR, "test_audio/fr")
TEST_PATH_ES = os.path.join(ROOT_DIR, "test_audio/es")
TEST_PATH_DE = os.path.join(ROOT_DIR, "test_audio/de")
TEST_PATH_IT = os.path.join(ROOT_DIR, "test_audio/it")

def transliteration(transcription, text, lang):
    transliterated = []
    translit_dict = {}
    if lang == 'pl':
        translit_dict = {'a': ['ą'], 'c': ['ć'], 'e': ['ę'], 'n': ['ń'], 'o': ['ó'], 's': ['ś'], 'z': ['ź', 'ż']}
    if lang == 'fr':
        translit_dict = {'c': ['ç'], 'e': ['é', 'ê', 'è', 'ë'], 'a': ['â', 'à'], 'i': ['î', 'ì', 'ï'],
                         'o': ['ô', 'ò'], 'u': ['û', 'ù', 'ü']}
    if lang == 'es':
        translit_dict = {'a': ['á'], 'i': ['í'], 'e': ['é'], 'n': ['ñ'], 'o': ['ó'], 'u': ['ú', 'ü']}
    if lang == 'de':
        translit_dict = {'a': ['ä'], 's': ['ß'], 'o': ['ö'], 'u': ['ú', 'ü']}
    transcription = re.sub("`|'|-", "", transcription)
    text = re.sub("`|'|-", "", text)
    if len(transcription.strip()) == len(text.strip()):
        for ind, letter in enumerate(transcription):
            if letter in translit_dict.keys():
                if letter != text[ind]:
                    for l in translit_dict[letter]:
                        if l == text[ind]:
                                transliterated.append(l)
                        else:
                            transliterated.append(letter)
                else:
                        transliterated.append(letter)
            else:
                    transliterated.append(letter)
        translit_str = ''.join(transliterated)
        if translit_str != '':
            error = wer(translit_str.strip(), text.strip())
            return error, translit_str, text
        else:
            error = wer(transcription.strip(), text.strip())
            return error, transcription, text
    else:
        error = wer(transcription.strip(), text.strip())
        return error, transcription, text

class TestGetSTT(unittest.TestCase):

    def test_en_stt(self):
        LOG.info("ENGLISH STT MODEL")
        stt = VoskKaldiSTT('en')
        LOG.info('Running inference.')
        for file in os.listdir(TEST_PATH_EN):
            inference_start = timer()
            transcription = ' '.join(file.split('_')[:-1]).lower()
            path = ROOT_DIR+'/test_audio/en/'+file
            text = stt.execute(path, language=None)
            error = wer(transcription.strip(), text[0].strip())
            LOG.info('Input: {}\nOutput:{}\nWER: {}'.format(transcription, text[0], error))
            inference_end = timer() - inference_start
            LOG.info('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, text[1]))
            self.assertTrue(error < 0.6)

    def test_fr_stt(self):
        LOG.info("FRENCH STT MODEL")
        stt = VoskKaldiSTT('fr')
        for file in os.listdir(TEST_PATH_FR):
            inference_start = timer()
            transcription = ' '.join(file.split('_')[:-1]).lower()
            path = ROOT_DIR + '/test_audio/fr/' + file
            text = stt.execute(path)
            result = transliteration(transcription, text[0], 'fr')
            LOG.info('Input: {}\nOutput:{}\nWER: {}'.format(result[1], result[2], result[0]))
            inference_end = timer() - inference_start
            LOG.info('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, text[1]))
            self.assertTrue(result[0] < 0.6)

    def test_es_stt(self):
        LOG.info("SPANISH STT MODEL")
        stt = VoskKaldiSTT('es')
        for file in os.listdir(TEST_PATH_ES):
            inference_start = timer()
            transcription = ' '.join(file.split('_')[:-1]).lower()
            path = ROOT_DIR + '/test_audio/es/' + file
            text = stt.execute(path)
            result = transliteration(transcription, text[0], 'es')
            LOG.info('Input: {}\nOutput:{}\nWER: {}'.format(result[1], result[2], result[0]))
            inference_end = timer() - inference_start
            LOG.info('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, text[1]))
            self.assertTrue(result[0] < 0.6)

    def test_de_stt(self):
        LOG.info("GERMAN STT MODEL")
        stt = VoskKaldiSTT('de')
        for file in os.listdir(TEST_PATH_DE):
            inference_start = timer()
            transcription = ' '.join(file.split('_')[:-1]).lower()
            path = ROOT_DIR + '/test_audio/de/' + file
            text = stt.execute(path)
            result = transliteration(transcription, text[0], 'de')
            LOG.info('Input: {}\nOutput:{}\nWER: {}'.format(result[1], result[2], result[0]))
            inference_end = timer() - inference_start
            LOG.info('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, text[1]))
            self.assertTrue(result[0] < 0.6)

    def test_it_stt(self):
        LOG.info("ITALIAN STT MODEL")
        stt = VoskKaldiSTT('it')
        for file in os.listdir(TEST_PATH_IT):
            inference_start = timer()
            transcription = ' '.join(file.split('_')[:-1]).lower()
            path = ROOT_DIR + '/test_audio/it/' + file
            text = stt.execute(path)
            result = transliteration(transcription, text[0], 'it')
            LOG.info('Input: {}\nOutput:{}\nWER: {}'.format(result[1], result[2], result[0]))
            inference_end = timer() - inference_start
            LOG.info('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, text[1]))
            self.assertTrue(result[0] < 0.6)


if __name__ == '__main__':
    unittest.main()
