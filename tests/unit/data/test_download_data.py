import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.data.download_data import download_data, main


class TestDownloadData:
    '''Test suite for download_data function.'''

    def test_download_data_success(self, tmp_path):
        '''Test successful data download with mocked download function.'''
        # Setup
        raw_data_path = str(tmp_path / 'raw_data')
        mock_file_path = str(tmp_path / 'source' / 'dataset.zip')
        os.makedirs(os.path.dirname(mock_file_path), exist_ok=True)
        Path(mock_file_path).touch()  # Create the mock file

        mock_download = MagicMock(return_value=mock_file_path)

        # Execute
        result = download_data(
            raw_data_path=raw_data_path,
            download_func=mock_download,
        )

        # Assert
        assert result == str(Path(raw_data_path) / 'dataset.zip')
        assert os.path.exists(result)
        assert os.path.isdir(raw_data_path)
        mock_download.assert_called_once_with()

    def test_download_data_with_different_file_types(self, tmp_path):
        '''Test download with different file types (csv instead of zip).'''
        # Setup
        raw_data_path = str(tmp_path / 'raw_data')
        mock_file_path = str(tmp_path / 'source' / 'custom_dataset.csv')
        os.makedirs(os.path.dirname(mock_file_path), exist_ok=True)
        Path(mock_file_path).touch()

        mock_download = MagicMock(return_value=mock_file_path)

        # Execute
        result = download_data(
            raw_data_path=raw_data_path,
            download_func=mock_download,
        )

        # Assert
        assert os.path.exists(result)
        assert result.endswith('custom_dataset.csv')
        mock_download.assert_called_once_with()

    def test_download_data_creates_directory(self, tmp_path):
        '''Test that download_data creates the destination directory if it doesn't exist.'''
        # Setup
        raw_data_path = str(tmp_path / 'nested' / 'path' / 'to' / 'raw_data')
        mock_file_path = str(tmp_path / 'source' / 'dataset.zip')
        os.makedirs(os.path.dirname(mock_file_path), exist_ok=True)
        Path(mock_file_path).touch()

        mock_download = MagicMock(return_value=mock_file_path)

        # Ensure directory doesn't exist yet
        assert not os.path.exists(raw_data_path)

        # Execute
        result = download_data(
            raw_data_path=raw_data_path,
            download_func=mock_download,
        )

        # Assert
        assert os.path.exists(raw_data_path)
        assert os.path.exists(result)

    def test_download_data_empty_raw_data_path_raises_error(self, tmp_path):
        '''Test that empty raw_data_path raises ValueError.'''
        mock_download = MagicMock()

        with pytest.raises(ValueError, match='raw_data_path cannot be empty'):
            download_data(raw_data_path='', download_func=mock_download)

        mock_download.assert_not_called()

    def test_download_data_none_download_func_raises_error(self, tmp_path):
        '''Test that None download_func raises ValueError.'''
        with pytest.raises(ValueError, match='download_func cannot be empty'):
            download_data(
                raw_data_path=str(tmp_path),
                download_func=None,
            )

    def test_download_data_file_exists_after_download(self, tmp_path):
        '''Test that the file is properly moved to the destination.'''
        # Setup
        raw_data_path = str(tmp_path / 'raw_data')
        source_dir = tmp_path / 'source'
        source_dir.mkdir()
        mock_file_path = str(source_dir / 'accidents.zip')
        Path(mock_file_path).write_text('test data')

        mock_download = MagicMock(return_value=mock_file_path)

        # Execute
        result = download_data(
            raw_data_path=raw_data_path,
            download_func=mock_download,
        )

        # Assert
        assert os.path.exists(result)
        assert not os.path.exists(mock_file_path)  # Original should be moved, not copied
        with open(result) as f:
            assert f.read() == 'test data'

    def test_download_data_preserves_filename(self, tmp_path):
        '''Test that the filename is preserved when moving to destination.'''
        # Setup
        raw_data_path = str(tmp_path / 'raw_data')
        source_file = tmp_path / 'source' / 'my_data.tar.gz'
        source_file.parent.mkdir(parents=True, exist_ok=True)
        source_file.touch()

        mock_download = MagicMock(return_value=str(source_file))

        # Execute
        result = download_data(
            raw_data_path=raw_data_path,
            download_func=mock_download,
        )

        # Assert
        assert result.endswith('my_data.tar.gz')
        assert os.path.exists(result)
        assert os.path.dirname(result) == raw_data_path


class TestMain:
    '''Test suite for main function.'''

    @patch('src.data.download_data.download_data')
    def test_main_calls_download_data(self, mock_download_data):
        '''Test that main function calls download_data with the correct path.'''
        # Setup
        mock_download_data.return_value = '/path/to/data'

        # Execute
        main()

        # Assert
        mock_download_data.assert_called_once()
        call_args = mock_download_data.call_args
        # First positional argument should be the raw_data_path
        raw_data_path = call_args[0][0]
        assert raw_data_path.endswith('data/raw/')
        # Second positional argument should be a callable download_func
        download_func = call_args[0][1]
        assert callable(download_func)

    @patch.dict(os.environ, {'RAW_DATA_PATH': 'custom/data/path'})
    @patch('src.data.download_data.download_data')
    def test_main_uses_env_variable(self, mock_download_data):
        '''Test that main function respects RAW_DATA_PATH environment variable.'''
        # Setup
        mock_download_data.return_value = '/path/to/data'

        # Execute
        main()

        # Assert
        mock_download_data.assert_called_once()
        call_args = mock_download_data.call_args
        raw_data_path = call_args[0][0]
        assert 'custom/data/path' in raw_data_path
