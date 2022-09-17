use io_utils::imux::IMuxSync;

#[inline]
pub fn serialize<W: std::io::Write + Send, T: ?Sized>(
    writer: &mut IMuxSync<W>,
    value: &T,
) -> Result<(), bincode::Error>
where
    T: serde::Serialize,
{
    // println!("{}",value);
    let bytes: Vec<u8> = bincode::serialize(value).unwrap();
    let res:std::os::raw::c_char = bincode::deserialize(&bytes[..]).unwrap();
    // println!("{}",res);
    let _ = writer.write(&bytes).expect("sending fail");
    // writer.flush()?;
    writer.flush().expect("could not flush");
    Ok(())
}


#[inline]
pub fn deserialize<R, T>(reader: &mut IMuxSync<R>) -> bincode::Result<T>
where
    R: std::io::Read + Send,
    T: serde::de::DeserializeOwned,
{
    // let bytes: Vec<u8> = reader.read()?;
    let bytes: Vec<u8> = reader.read().expect("could not read");
    bincode::deserialize(&bytes[..])
}
