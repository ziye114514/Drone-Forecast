"""Official SDK delta-time iterator for paced replay, keeping native timestamps."""
def join_native(pieces,dtype):
    """Copy canonical packed event bytes, without structured field casting."""
    import numpy as np
    if not pieces:return np.empty(0,dtype)
    result=np.empty(sum(len(p) for p in pieces),dtype)
    target=result.view(np.uint8);offset=0
    for piece in pieces:
        if piece.dtype!=dtype:raise TypeError('Cannot reinterpret incompatible event layout')
        source=np.ascontiguousarray(piece).view(np.uint8).reshape(-1)
        target[offset:offset+source.nbytes]=source;offset+=source.nbytes
    return result


def native_batches(raw, begin, duration,build_index=True,step_us=20000,read_chunk=100000,read_buffers=3,reader_backend='iterator',byte_transport=True):
    from metavision_core.event_io import EventsIterator
    if not isinstance(step_us,int) or step_us<=0:raise ValueError('Positive integer step_us required')
    if not isinstance(read_chunk,int) or read_chunk<=0:raise ValueError('Positive integer read_chunk required')
    if not isinstance(read_buffers,int) or read_buffers<=0:raise ValueError('Positive integer read_buffers required')
    if reader_backend not in ('iterator','rolling','cooperative'):raise ValueError('Unknown native reader backend')
    if reader_backend=='rolling':
        from metavision_hal import DeviceDiscovery,RawFileConfig
        from metavision_core.event_io import RawReader
        config=RawFileConfig();config.do_time_shifting=True;config.n_events_to_read=read_chunk
        config.n_read_buffers=read_buffers;config.build_index=False
        device=DeviceDiscovery.open_raw_file(str(raw),config)
        if device is None:raise OSError('Cannot open RAW for public rolling reader')
        reader=RawReader.from_device(device,max_events=8000000)
        if tuple(reader.get_size())!=(720,1280):raise ValueError('Expected1280x720 RAW')
        start=round(begin*1e6);end=start+round(duration*1e6)
        with reader:
            reader.seek_time(start)
            while reader.current_time<end and not reader.is_done():
                events=reader.load_delta_t(min(step_us,end-reader.current_time))
                current=reader.current_time
                assert not len(events) or (events['t'].min()>=current-step_us and events['t'].max()<current)
                yield current/1e6,events
        return
    start=round(begin*1e6);end=start+round(duration*1e6)
    aligned=start-start%step_us
    # EventsIterator requires global-grid starts. Keep the caller's original
    # cutoffs by reassembling two adjacent native batches, without altering
    # timestamps, dropping events, or changing any SDK code.
    options=dict(start_ts=aligned,delta_t=step_us,max_duration=end-aligned+step_us,
                 mode='delta_t',relative_timestamps=False)
    if build_index and reader_backend!='cooperative':
        iterator=EventsIterator(input_path=str(raw),**options)
    else:
        # Match installed raw_reader.initiate_device's time shift and read
        # chunk. Its sequential seek does not use HAL's background index.
        # Public SDK configuration only; never patch the reader or decoder.
        from metavision_hal import DeviceDiscovery,RawFileConfig
        config=RawFileConfig();config.do_time_shifting=True
        config.n_events_to_read=read_chunk;config.build_index=False
        config.n_read_buffers=read_buffers
        device=DeviceDiscovery.open_raw_file(str(raw),config)
        if device is None:raise OSError('Cannot open RAW with explicit non-indexed configuration')
        if reader_backend=='cooperative':
            from cooperative_raw_reader import CooperativeEventsIterator
            iterator=CooperativeEventsIterator.from_device(device,**options)
        else:iterator=EventsIterator.from_device(device,**options)
    if tuple(iterator.get_size())!=(720,1280):raise ValueError('Expected1280x720 RAW')
    import numpy as np
    cutoff=min(start+step_us,end);pieces=[]
    for events in iterator:
        grid_end=round(iterator.get_current_time())
        if len(events):
            # SDK events are timestamp ordered. Avoid copying the entire
            # strided timestamp column for trivial outside-range searches.
            first=0 if events['t'][0]>=start else int(np.searchsorted(events['t'],start))
            last=len(events) if events['t'][-1]<end else int(np.searchsorted(events['t'],end))
            remaining=events[first:last]
        else:remaining=events
        while cutoff<=grid_end and cutoff<=end:
            split=int(np.searchsorted(remaining['t'],cutoff))
            if split:pieces.append(remaining[:split])
            # NumPy's inferred concatenation dtype repacks EventCD and removes
            # its two padding bytes. Downstream shared slots use the canonical
            # SDK16-byte ABI, so request that exact dtype explicitly.
            batch=join_native(pieces,events.dtype) if byte_transport else np.concatenate(pieces,dtype=events.dtype) if pieces else np.empty(0,events.dtype)
            assert not len(batch) or (batch['t'][0]>=max(start,cutoff-step_us) and batch['t'][-1]<cutoff)
            yield cutoff/1e6,batch
            pieces=[];remaining=remaining[split:]
            if cutoff==end:return
            cutoff=min(cutoff+step_us,end)
        if len(remaining):pieces.append(join_native([remaining],events.dtype) if byte_transport else remaining.copy())
